import os
import json
import time
import pandas as pd
import re
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq, RateLimitError


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=0
)


MODEL_NAME = "llama-3.1-8b-instant"


SENTIMENT_COLUMNS = [
    "llm_positive",
    "llm_negative",
    "llm_neutral",
    "llm_label",
    "llm_confidence",
    "llm_score",
    "llm_relevance",
    "llm_impact_magnitude",
    "llm_uncertainty",
]


class GroqDailyLimitReached(Exception):
    pass


def build_prompt(ticker: str, news_text: str) -> str:
    return f"""
You are a financial news sentiment classifier.

Analyze the news from the perspective of ticker: {ticker}

News text:
\"\"\"{news_text}\"\"\"

Return ONLY valid JSON with exactly these fields:

{{
  "llm_positive": float between 0 and 1 with 4 decimal points,
  "llm_negative": float between 0 and 1 with 4 decimal points,
  "llm_neutral": float between 0 and 1 with 4 decimal points,
  "llm_label": "positive" or "negative" or "neutral",
  "llm_confidence": float between 0 and 1 with 4 decimal points,
  "llm_score": float between -1 and 1 with 4 decimal points,
  "llm_relevance": float between 0 and 1 with 4 decimal points,
  "llm_impact_magnitude": float between 0 and 1 with 4 decimal points,
  "llm_uncertainty": float between 0 and 1 with 4 decimal points
}}

Meaning:
- llm_positive, llm_negative, llm_neutral: class probabilities. They should roughly sum to 1.
- llm_label: final sentiment label.
- llm_confidence: confidence in the final label.
- llm_score: negative means bearish, positive means bullish, 0 means neutral.
- llm_relevance: how relevant this news is to the ticker.
- llm_impact_magnitude: expected market impact strength, not direction.
- llm_uncertainty: uncertainty caused by vague, speculative, incomplete, or conflicting information.

Important rules:
- Judge sentiment only from the perspective of {ticker}.
- If the news is not relevant to {ticker}, set llm_relevance low.
- If the news mentions the company but has no clear financial impact, use neutral.
- Do not add explanation.
- Return JSON only.
""".strip()


def is_daily_limit_error(error: Exception) -> bool:
    """
    Detects whether Groq 429 error is likely a daily limit:
    - requests per day
    - tokens per day

    Groq limit types include RPD and TPD, so once these are hit,
    the script should stop instead of retrying.
    """
    error_text = str(error).lower()

    daily_limit_keywords = [
        "requests per day",
        "request per day",
        "rpd",
        "tokens per day",
        "token per day",
        "tpd",
        "daily",
    ]

    return any(keyword in error_text for keyword in daily_limit_keywords)


def parse_llm_json(content: str) -> dict:
    """
    Safely parse model JSON.
    Sometimes models may still add tiny extra text, so this tries direct parse first,
    then extracts the first JSON object.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in model output: {content}")

        return json.loads(content[start:end])


def validate_result(result: dict) -> dict:
    """
    Ensures all required columns exist and values are in reasonable ranges.
    """
    clean = {}

    clean["llm_positive"] = float(result.get("llm_positive", 0.0))
    clean["llm_negative"] = float(result.get("llm_negative", 0.0))
    clean["llm_neutral"] = float(result.get("llm_neutral", 1.0))

    label = str(result.get("llm_label", "neutral")).lower().strip()
    if label not in ["positive", "negative", "neutral"]:
        label = "neutral"
    clean["llm_label"] = label

    clean["llm_confidence"] = float(result.get("llm_confidence", 0.0))
    clean["llm_score"] = float(result.get("llm_score", 0.0))
    clean["llm_relevance"] = float(result.get("llm_relevance", 0.0))
    clean["llm_impact_magnitude"] = float(result.get("llm_impact_magnitude", 0.0))
    clean["llm_uncertainty"] = float(result.get("llm_uncertainty", 1.0))

    # Clamp values
    for col in [
        "llm_positive",
        "llm_negative",
        "llm_neutral",
        "llm_confidence",
        "llm_relevance",
        "llm_impact_magnitude",
        "llm_uncertainty",
    ]:
        clean[col] = max(0.0, min(1.0, clean[col]))

    clean["llm_score"] = max(-1.0, min(1.0, clean["llm_score"]))

    return clean


def analyse_news_with_groq(ticker: str, news_text: str, max_retries: int = 5) -> dict:
    if pd.isna(news_text) or str(news_text).strip() == "":
        return {
            "llm_positive": 0.0,
            "llm_negative": 0.0,
            "llm_neutral": 1.0,
            "llm_label": "neutral",
            "llm_confidence": 1.0,
            "llm_score": 0.0,
            "llm_relevance": 0.0,
            "llm_impact_magnitude": 0.0,
            "llm_uncertainty": 0.0,
        }

    prompt = build_prompt(ticker=ticker, news_text=str(news_text))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict financial sentiment analysis engine. Return JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json(content)

            return validate_result(result)

        except RateLimitError as e:
            # Daily limit: stop whole CSV processing
            if is_daily_limit_error(e):
                raise GroqDailyLimitReached(f"Groq daily limit reached: {e}")

            # Per-minute limit: retry with backoff
            wait_seconds = min(2 ** attempt, 60)
            print(f"Groq per-minute rate limit hit. Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

        except Exception as e:
            wait_seconds = min(2 ** attempt, 60)
            print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    # Fallback only for normal failed rows, not daily limit
    return {
        "llm_positive": 0.0,
        "llm_negative": 0.0,
        "llm_neutral": 1.0,
        "llm_label": "neutral",
        "llm_confidence": 0.0,
        "llm_score": 0.0,
        "llm_relevance": 0.0,
        "llm_impact_magnitude": 0.0,
        "llm_uncertainty": 1.0,
    }


def generate_llm_sentiment(
    input_path: str,
    output_path: str,
    batch_size: int = 100,
    resume: bool = True,
):
    df = pd.read_csv(input_path)

    required_cols = ["ticker", "news_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # If output already exists, resume from it
    if resume and os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        df_existing = pd.read_csv(output_path)

        # Make sure existing file has the sentiment columns
        for col in SENTIMENT_COLUMNS:
            if col not in df_existing.columns:
                df_existing[col] = None

        df = df_existing
    else:
        for col in SENTIMENT_COLUMNS:
            if col not in df.columns:
                df[col] = None

    pending_mask = df["llm_label"].isna()
    pending_indices = df[pending_mask].index.tolist()

    print(f"Total rows: {len(df)}")
    print(f"Pending rows: {len(pending_indices)}")

    processed_since_save = 0

    for idx in tqdm(pending_indices, desc="Generating Groq LLM sentiment"):
        ticker = str(df.at[idx, "ticker"])
        news_text = df.at[idx, "news_text"]

        try:
            result = analyse_news_with_groq(
                ticker=ticker,
                news_text=news_text,
            )

        except GroqDailyLimitReached as e:
            print("\nDaily Groq limit reached.")
            print(str(e))
            print(f"Saving progress to: {output_path}")

            df.to_csv(output_path, index=False)

            print("Stopped safely. You can rerun the script tomorrow with resume=True.")
            return

        for col in SENTIMENT_COLUMNS:
            df.at[idx, col] = result[col]

        processed_since_save += 1

        if processed_since_save >= batch_size:
            df.to_csv(output_path, index=False)
            processed_since_save = 0

    df.to_csv(output_path, index=False)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    ticker = "nvda"
    generate_llm_sentiment(
        input_path=f"../data/cleaned_merged_{ticker}_news.csv",
        output_path=f"../data/groq_{ticker}_news_sentiment.csv",
        batch_size=100,
        resume=True,
    )
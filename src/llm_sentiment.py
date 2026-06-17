import ollama
import pandas as pd
import os
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

class LLMRawSentiment(BaseModel):
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)

    relevance: float = Field(ge=0.0, le=1.0)
    impact_magnitude: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self):
        total = self.positive + self.negative + self.neutral

        if abs(total - 1.0) > 0.02:
            raise ValueError(
                "positive, negative, and neutral must sum to approximately 1"
            )

        return self


def analyse_news_with_llm(summary, ticker, model = "llama3.2:latest"):
    prompt = f"""
Analyse the following financial news summary from the perspective of
the specified stock ticker.

Ticker: {ticker}
Summary: {summary}

Return:

- positive: value from 0 to 1
- negative: value from 0 to 1
- neutral: value from 0 to 1

The three sentiment values must sum to 1.

Also return:

- relevance:
  0 means unrelated to the ticker.
  1 means directly relevant to the ticker.

- impact_magnitude:
  0 means negligible expected impact.
  1 means extremely significant expected impact.

- uncertainty:
  0 means the interpretation is clear.
  1 means the interpretation is highly ambiguous.

Judge sentiment based on the likely effect on the named company's stock,
not merely the emotional tone of the summary.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial news sentiment classifier. "
                    "Return only valid structured data matching the schema."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format=LLMRawSentiment.model_json_schema(),
        options={
            "temperature": 0,
            "seed": 42,
            "num_predict": 100,
        },
        keep_alive="30m",
    )

    result = LLMRawSentiment.model_validate_json(
        response.message.content
    )

    probabilities = {
        "positive": result.positive,
        "negative": result.negative,
        "neutral": result.neutral,
    }

    label = max(probabilities, key=probabilities.get)

    return {
        "llm_positive": result.positive,
        "llm_negative": result.negative,
        "llm_neutral": result.neutral,
        "llm_label": label,
        "llm_confidence": probabilities[label],
        "llm_score": result.positive - result.negative,
        "llm_relevance": result.relevance,
        "llm_impact_magnitude": result.impact_magnitude,
        "llm_uncertainty": result.uncertainty,
    }

LLM_OUTPUT_COLUMNS = [
    "llm_positive",
    "llm_negative",
    "llm_neutral",
    "llm_label",
    "llm_confidence",
    "llm_score",
    "llm_relevance",
    "llm_impact_magnitude",
    "llm_uncertainty",
    "llm_error",
]

def generate_llm_sentiment(input_path, output_path, ticker, batch_size = 100):
    df = pd.read_csv(input_path, parse_dates=["pub_time"])
    news_list = df["news_text"].tolist()
    llm_results_buffer = []

    # Determine where to resume.
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        start_idx = len(existing_df)

        print(f"Resuming from row {start_idx:,}")
    else:
        start_idx = 0

    remaining_df = df.iloc[start_idx:]
    progress_bar = tqdm(enumerate(news_list, start=start_idx+1), total=len(news_list), initial=start_idx, desc="Generating LLM sentiment")

    for count, news in progress_bar:
        result = analyse_news_with_llm(news, ticker)
        llm_results_buffer.append(result)

        if len(llm_results_buffer) == batch_size or count == len(df):
            # Calculate how many items are in this specific batch
            current_batch_size = len(llm_results_buffer)

            # Slice the matching rows from the original dataframe
            batch_start_idx = count - current_batch_size
            df_slice = df.iloc[batch_start_idx:count].reset_index(drop=True)

            # Turn the list of LLM dictionaries into a DataFrame
            llm_df = pd.DataFrame(llm_results_buffer)

            # Combine original columns and new LLM columns side-by-side
            batch_combined = pd.concat([df_slice, llm_df], axis=1)

            # 4. Save cleanly to CSV using append mode ('a')
            # Only write the header if the file does not exist yet
            file_exists = os.path.exists(output_path)
            batch_combined.to_csv(output_path, mode='a', index=False, header=not file_exists)

            # Clear the buffer for the next batch of 100
            llm_results_buffer = []

    print(f"Results saved to CSV.")



ticker = "NVDA"
input_path = f"../data/cleaned_merged_{ticker.lower()}_news.csv"
output_path = f"../data/llm_{ticker.lower()}_news_sentiment.csv"
generate_llm_sentiment(input_path, output_path, ticker)

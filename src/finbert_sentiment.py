import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_news_text(df):
    """
    Combine title and summary.

    Rules:
    - Missing title becomes an empty string.
    - Missing summary is allowed.
    - If summary is missing, only the title is used.
    - Duplicate whitespace is removed.
    """
    title = df["title"].fillna("").astype(str)
    summary = df["summary"].fillna("").astype(str)

    text = np.where(
        summary.ne(""),             # Condition: summary not empty
        title + ". " + summary,     # If true: title + summary
        title,                      # If false: title
    )

    return pd.Series(text, index=df.index)


def generate_finbert_sentiment(df, batch_size = 32, max_length = 512):
    """
    Generate FinBERT sentiment probabilities for every row.

    Added columns:
    - finbert_positive
    - finbert_negative
    - finbert_neutral
    - finbert_label
    - finbert_confidence
    - finbert_score

    finbert_score = positive_probability - negative_probability
    Its range is approximately [-1, 1].
    """
    required_columns = {"title", "summary"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy()
    result["news_text"] = build_news_text(result)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    # Read the label mapping from the model itself instead of assuming
    # that label 0, 1, and 2 always have a particular meaning.
    id_to_label = {
        int(label_id): label.lower()
        for label_id, label in model.config.id2label.items()
    }

    all_probabilities = []

    texts = result["news_text"].tolist()

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Running FinBERT",
    ):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    probability_array = np.concatenate(
        all_probabilities,
        axis=0,
    )

    # Convert model output IDs to named probability columns.
    for label_id, label_name in id_to_label.items():
        result[f"finbert_{label_name}"] = (
            probability_array[:, label_id]
        )

    predicted_ids = probability_array.argmax(axis=1)

    result["finbert_label"] = [
        id_to_label[int(label_id)]
        for label_id in predicted_ids
    ]

    result["finbert_confidence"] = probability_array.max(axis=1)

    # Continuous directional sentiment score:
    # positive values = more positive
    # negative values = more negative
    # values near zero = neutral or uncertain
    result["finbert_score"] = (
        result["finbert_positive"]
        - result["finbert_negative"]
    )

    return result


def process_news_csv(
    input_path, output_path, batch_size = 32):
    df = pd.read_csv(input_path, parse_dates=["pub_time"])

    result = generate_finbert_sentiment(
        df=df,
        batch_size=batch_size,
        max_length=512,
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved {len(result):,} rows to: {output_path}")


ticker = "nvda"
process_news_csv(
    input_path=f"../data/cleaned_merged_{ticker}_news.csv",
    output_path=f"../data/final_{ticker}_news_sentiment.csv",
    batch_size=32
)
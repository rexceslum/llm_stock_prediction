import pandas as pd
import os
from datetime import datetime
from massive import RESTClient

client = RESTClient("gJJ78yonpeHHCdVLv_kkNvqkzxk5etNx")
ticker = "NVDA"
mapping = {
    "very positive": 1,
    "positive": 0.8,
    "neutral/positive": 0.3,
    "neutral": 0,
    "mixed": 0,
    "hold": 0,
    "neutral/negative": -0.3,
    "negative": -1
}

news = []
for n in client.list_ticker_news(
        ticker="NVDA",
        published_utc_gte="2023-05-01T00:00:00Z",
        published_utc_lt="2024-01-01T00:00:00Z",
        # published_utc="2024-05-01T08:00:00Z",
        order="desc",
        limit=1000,
        sort="published_utc",
	):
    # print(n)
    sentiment = None
    sentiment_reasoning = None

    for insight in n.insights or []:
        if insight.ticker == ticker:
            sentiment = insight.sentiment
            sentiment_reasoning = insight.sentiment_reasoning
            break

    news.append({
        "pub_time": datetime.strptime(n.published_utc,"%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "title": n.title,
        "summary": n.description,
        "sentiment_score": mapping[sentiment.lower()] if sentiment else None,
    })

df = pd.DataFrame(news)
df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce")
csv_file = f"../data/massive_{ticker.lower()}_news.csv"

if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file, parse_dates=["pub_time"])
    existing_df["pub_time"] = pd.to_datetime(existing_df["pub_time"], errors="coerce")
    combined_df = pd.concat([existing_df, df])
    combined_df.drop_duplicates(subset=["pub_time", "ticker", "title"], inplace=True)
    combined_df.sort_values(by="pub_time", ascending=True, inplace=True)
    combined_df["pub_time"] = combined_df["pub_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined_df.to_csv(csv_file, index=False)
else:
    df.sort_values(by="pub_time", ascending=True, inplace=True)
    df["pub_time"] = df["pub_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(csv_file, index=False)

print(df)
print(f"\nSaved articles to csv")
print(f"Total articles: {len(news)}")

import requests
import pandas as pd
import os
from datetime import datetime

api_key = "6a242c8a06a730.47874144"
ticker="NVDA"
url = "https://eodhd.com/api/news"
params = {
    "api_token": api_key,
    "s": ticker+".US",
    "from": "2023-05-01",
    "to": "2026-04-30",
    "limit": 1000,
    "fmt": "json",
}
responses = requests.get(url, params=params)
data = responses.json()
print(data)

rows = []
for article in data:
    title = article.get("title", "")
    pub_time = datetime.strptime(article.get("date", ""), "%Y-%m-%dT%H:%M:%S%z")
    summary = article.get("content", "")
    topics = article.get("tags", [])
    sentiment_score = article.get("sentiment").get("polarity", "")
    print(f"Title: {title}")
    print(f"Pub Time: {pub_time}")
    print(f"Summary: {summary}")
    print(f"Topics: {topics}")
    print(f"Sentiment score: {article.get('sentiment')}")

    rows.append({
        "pub_time": pub_time,
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "sentiment_score": sentiment_score
    })

print(f"Total articles: {len(rows)}")

csv_file = f"../data/eodhd_{ticker.lower()}_news.csv"
df = pd.DataFrame(rows)
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file, parse_dates=["pub_time"])
    combined_df = pd.concat([existing_df, df])
    combined_df.drop_duplicates(subset=["pub_time", "ticker", "title"], inplace=True)
    combined_df.sort_values(by="pub_time", ascending=True, inplace=True)
    combined_df.to_csv(csv_file, index=False)
else:
    df.sort_values(by="pub_time", ascending=True, inplace=True)
    df.to_csv(csv_file, index=False)
print(f"Saved articles to csv")
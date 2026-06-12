import requests
import pandas as pd
import os
from datetime import datetime

api_key = "&api_token=eA4Mbxm4D6Tvosi8b9uvwqhjZOTdybn4e0451sMc"
ticker="NVDA"
tickers = f"&symbols={ticker}"
other_config = "&filter_entities=true&language=en"
time_from = "&published_after=2023-05-01T00:00:00"
time_to = "&published_before=2026-05-01T00:00:00"
sort = "&sort=published_at"
sort_order = "&sort_order=desc"
limit = "&limit=1000"
url = "https://api.marketaux.com/v1/news/all?"+api_key+other_config+tickers+time_from+time_to+sort+sort_order+limit
responses = requests.get(url)
data = responses.json()
print(f"Metadata: {data.get('meta')}\n")

rows = []
articles = data.get("data", [])
for article in articles:
    title = article.get("title", "")
    pub_time = datetime.strptime(article.get("published_at", ""), "%Y-%m-%dT%H:%M:%S.%fZ")
    summary = article.get("description", "")
    relevance_score = article.get("relevance_score", "")
    match_score = article.get("match_score", "")
    sentiment_score = article.get("entities", []).get("sentiment_score")
    print(f"Title: {title}")
    print(f"Pub Time: {pub_time}")
    print(f"Summary: {summary}")
    print(f"Relevance score: {relevance_score}")
    print(f"Match score: {match_score}")
    print(f"Sentiment score: {sentiment_score}\n")

    rows.append({
        "pub_time": pub_time,
        "title": title,
        "summary": summary,
        "sentiment_score": sentiment_score
    })

print(f"Total articles retrieved: {data.get('meta').get('returned')}")

csv_file = "../data/marketaux_nvda_news.csv"
df = pd.DataFrame(rows)
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file, parse_dates=["pub_time"])
    combined_df = pd.concat([existing_df, df])
    combined_df.drop_duplicates(subset=["pub_time", "title"], inplace=True)
    combined_df.sort_values(by="pub_time", ascending=True, inplace=True)
    combined_df.to_csv(csv_file, index=False)
else:
    df.sort_values(by="pub_time", ascending=True, inplace=True)
    df.to_csv(csv_file, index=False)
print(f"Saved articles to csv")
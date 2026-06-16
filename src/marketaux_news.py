import requests
import pandas as pd
import os
import math
from datetime import datetime

# api_key = "eA4Mbxm4D6Tvosi8b9uvwqhjZOTdybn4e0451sMc"
api_key = "2L38mVttLE8cESKciOGlbgG58ToarfKznlVBrXlp"
ticker="MSFT"
page = 1
total_pages = 1
url = "https://api.marketaux.com/v1/news/all"
params = {
    "api_token": api_key,
    "symbols": ticker,
    "language": "en",
    "filter_entities": "true",
    "published_after": "2023-06-25T00:00:00",
    "published_before": "2023-06-26T00:00:00",
    "sort": "published_at",
    "sort_order": "desc",
    "limit": 1000,
    "page": page,
}

while page <= total_pages:
    responses = requests.get(url, params=params)
    data = responses.json()
    print(data)

    # Set pagination loop
    total_pages = math.ceil(data.get('meta').get('found') / 3)

    rows = []
    articles = data.get("data", [])
    for article in articles:
        title = article.get("title", "")
        pub_time = datetime.strptime(article.get("published_at", ""), "%Y-%m-%dT%H:%M:%S.%fZ")
        summary = article.get("description", "")
        relevance_score = article.get("relevance_score", "")
        entities = article.get("entities", [])
        match_score = entities[0].get("match_score", "")
        sentiment_score = entities[0].get("sentiment_score", "")
        print(f"Title: {title}")
        print(f"Pub Time: {pub_time}")
        print(f"Summary: {summary}")
        print(f"Relevance score: {relevance_score}")
        print(f"Match score: {match_score}")
        print(f"Sentiment score: {sentiment_score}\n")

        rows.append({
            "pub_time": pub_time,
            "ticker": ticker,
            "title": title,
            "summary": summary,
            "sentiment_score": sentiment_score
        })

    print(f"Total articles found: {data.get('meta').get('found')}")
    print(f"Total articles retrieved: {data.get('meta').get('returned')}")

    csv_file = f"../data/marketaux_{ticker.lower()}_news.csv"
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
    page += 1
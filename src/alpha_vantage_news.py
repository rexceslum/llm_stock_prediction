import requests
import pandas as pd
import os
from datetime import datetime

# api_key = "&apikey=4NH3GXMDDOMJKK6U"
api_key = "&apikey=89Y6O2Y8O8X77DCD"
ticker="MSFT"
tickers = f"&tickers={ticker}"
time_from = "&time_from=20230501T0000"
time_to = f"&time_to={datetime.strptime('2026-05-01 00:00', '%Y-%m-%d %H:%M').strftime('%Y%m%dT%H%M')}"
sort = "&sort=LATEST"
limit = "&limit=1000"
url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT"+api_key+tickers+time_from+time_to+sort+limit
responses = requests.get(url)
data = responses.json()
print(f"Sentiment score definition: {data.get('sentiment_score_definition')}")
print(f"Relevance score definition: {data.get('relevance_score_definition')}\n")

rows = []
articles = data.get("feed", [])
for article in articles:
    title = article.get("title", "")
    pub_time = datetime.strptime(article.get("time_published", ""), "%Y%m%dT%H%M%S")
    summary = article.get("summary", "")
    topics = article.get("topics", "")
    ticker_sentiment = article.get("ticker_sentiment", "")
    sentiment_score = article.get("overall_sentiment_score", "")
    sentiment_label = article.get("overall_sentiment_label", "")
    print(f"Title: {title}")
    print(f"Pub Time: {pub_time}")
    print(f"Summary: {summary}")
    print(f"Topics: {topics}")
    print(f"Ticker Sentiment: {ticker_sentiment}")
    print(f"Sentiment score: {sentiment_score}")
    print(f"Sentiment label: {sentiment_label}\n")

    rows.append({
        "pub_time": pub_time,
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "sentiment_score": sentiment_score
    })

print(f"Total articles: {data.get('items')}")

csv_file = f"../data/alpha_{ticker.lower()}_news.csv"
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
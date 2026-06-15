import requests
import pandas as pd
import os
from datetime import datetime

api_key = "&apikey=4NH3GXMDDOMJKK6U"
ticker="NVDA"
url = "https://financialmodelingprep.com/stable/news/stock-latest"
params = {
    "apikey": "azx2OPmgS6fs4qjyOyIhPOci6Emik3nc",
    "from": "2023-05-01",
    "to": "2026-04-30",
    "limit": 250,
}
responses = requests.get(url, params=params)
print(responses.content)
# data = responses.json()

# rows = []
# articles = data.get("feed", [])
# for article in articles:
#     title = article.get("title", "")
#     pub_time = datetime.strptime(article.get("time_published", ""), "%Y%m%dT%H%M%S")
#     summary = article.get("summary", "")
#     topics = article.get("topics", "")
#     ticker_sentiment = article.get("ticker_sentiment", "")
#     sentiment_score = article.get("overall_sentiment_score", "")
#     sentiment_label = article.get("overall_sentiment_label", "")
#     print(f"Title: {title}")
#     print(f"Pub Time: {pub_time}")
#     print(f"Summary: {summary}")
#     print(f"Topics: {topics}")
#     print(f"Ticker Sentiment: {ticker_sentiment}")
#     print(f"Sentiment score: {sentiment_score}")
#     print(f"Sentiment label: {sentiment_label}\n")
#
#     rows.append({
#         "pub_time": pub_time,
#         "ticker": ticker,
#         "title": title,
#         "summary": summary,
#         "sentiment_score": sentiment_score
#     })
#
# print(f"Total articles: {data.get('items')}")
#
# csv_file = f"../data/alpha_{ticker.lower()}_news.csv"
# df = pd.DataFrame(rows)
# if os.path.exists(csv_file):
#     existing_df = pd.read_csv(csv_file, parse_dates=["pub_time"])
#     combined_df = pd.concat([existing_df, df])
#     combined_df.drop_duplicates(subset=["pub_time", "ticker", "title"], inplace=True)
#     combined_df.sort_values(by="pub_time", ascending=True, inplace=True)
#     combined_df.to_csv(csv_file, index=False)
# else:
#     df.sort_values(by="pub_time", ascending=True, inplace=True)
#     df.to_csv(csv_file, index=False)
# print(f"Saved articles to csv")
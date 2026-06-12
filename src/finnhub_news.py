import finnhub
import pandas as pd
import os
import time
from datetime import datetime, timezone, timedelta

finnhub_client = finnhub.Client(api_key="d8hvd2pr01qgevbk502gd8hvd2pr01qgevbk5030")
ticker = "NVDA"
start_date = datetime(2023, 5, 1)
end_date = datetime(2025, 6, 16)
current = end_date
retry_count = 0
no_data_count = 0

while current >= start_date:
    if retry_count > 5:
        break
    if no_data_count > 7:
        break

    previous_day = current - timedelta(days=1)
    from_date = current.strftime("%Y-%m-%d")
    to_date = current.strftime("%Y-%m-%d")

    print(f"Fetching {ticker}: {from_date} -> {to_date}")
    try:
        articles = finnhub_client.company_news(ticker, _from=from_date, to=to_date)

        rows = []
        for article in articles or []:
            category = article.get("category", "")
            rows.append({
                "pub_time": datetime.fromtimestamp(article.get("datetime", ""), tz=timezone.utc),
                "ticker": ticker,
                "title": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "sentiment_score": None
            })

        print(f"\nTotal articles: {len(rows)}")

        if len(rows) > 0:
            csv_file = f"../data/finnhub_{ticker.lower()}_news.csv"
            df = pd.DataFrame(rows)
            df["pub_time"] = df["pub_time"].dt.tz_localize(None)
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
            no_data_count = 0
        else:
            no_data_count += 1

        current = previous_day
        retry_count = 0

    except Exception as e:
        print(f"Failed {from_date}: {e}")
        print("Retrying in 60 seconds...")
        retry_count += 1
        time.sleep(60)




# articles = finnhub_client.company_news(ticker, _from="2026-04-30", to="2026-04-30")
# print(articles)
#
# rows = []
# for article in articles:
#     title = article.get("headline", "")
#     pub_time = datetime.fromtimestamp(article.get("datetime", ""), tz=timezone.utc)
#     summary = article.get("summary", "")
#     category = article.get("category", "")
#     sentiment_score = None
#     rows.append({
#         "pub_time": pub_time,
#         "ticker": ticker,
#         "title": title,
#         "summary": summary,
#         "sentiment_score": sentiment_score
#     })
#
# print(f"\nTotal articles: {len(rows)}")
#
# csv_file = f"../data/finnhub_{ticker.lower()}_news.csv"
# df = pd.DataFrame(rows)
# if os.path.exists(csv_file):
#     existing_df = pd.read_csv(csv_file, parse_dates=["pub_time"])
#     df["pub_time"] = df["pub_time"].dt.tz_localize(None)
#     combined_df = pd.concat([existing_df, df])
#     combined_df.drop_duplicates(subset=["pub_time", "ticker", "title"], inplace=True)
#     combined_df.sort_values(by="pub_time", ascending=True, inplace=True)
#     combined_df["pub_time"] = (combined_df["pub_time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
#     combined_df.to_csv(csv_file, index=False)
# else:
#     df.sort_values(by="pub_time", ascending=True, inplace=True)
#     df["pub_time"] = df["pub_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
#     df.to_csv(csv_file, index=False)
# print(f"Saved articles to csv")
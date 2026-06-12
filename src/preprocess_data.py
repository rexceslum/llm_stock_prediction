import pandas as pd
from datetime import datetime

def chop_by_period(csv_file, start_date, end_date):
    data = pd.read_csv(csv_file, parse_dates=["pub_time"])
    print(f"Original rows: {len(data)}")
    data = data[
        (data["pub_time"] >= pd.Timestamp(start_date)) &
        (data["pub_time"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        ]
    data.to_csv(csv_file, index=False)
    print(f"Remaining rows: {len(data)}")

def find_missing_dates(csv_file, start_date, end_date):
    data = pd.read_csv(csv_file, parse_dates=["pub_time"])
    existing_dates = set(data["pub_time"].dt.date)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D").date
    missing_dates = sorted(set(all_dates) - existing_dates)

    print(f"Missing dates: {len(missing_dates)}")
    for d in missing_dates:
        print(d)



alpha_news = "../data/alpha_nvda_news.csv"
finnhub_news = "../data/finnhub_nvda_news.csv"
massive_news = "../data/massive_nvda_news.csv"

# chop_by_period("../data/finnhub_nvda_news.csv", "2023-05-01","2026-04-30")

print(f"Processing file: {alpha_news}")
find_missing_dates(alpha_news, "2023-05-01","2026-04-30")
print(f"\nProcessing file: {finnhub_news}")
find_missing_dates(finnhub_news, "2023-05-01","2026-04-30")
print(f"\nProcessing file: {massive_news}")
find_missing_dates(massive_news, "2023-05-01","2026-04-30")
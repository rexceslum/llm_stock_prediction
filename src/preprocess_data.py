import pandas as pd
from sqlalchemy import create_engine
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

def add_column(csv_file, column_name, value):
    df = pd.read_csv(csv_file)

    # Insert ticker as the second column (index 1)
    if column_name not in df.columns:
        df.insert(1, column_name, value)
    else:
        df[column_name] = value

    df.to_csv(csv_file, index=False)
    print(f"Added column: {column_name}")

def merge_news_csv(files, output_file=None):
    dfs = []

    # Read all csv files
    for file in files:
        df = pd.read_csv(file, parse_dates=["pub_time"])
        dfs.append(df)

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates
    merged_df.drop_duplicates(
        subset=["pub_time", "ticker", "title"],
        inplace=True
    )

    # Sort by publication time
    merged_df.sort_values(
        by="pub_time",
        ascending=True,
        inplace=True
    )

    # Format datetime for csv
    merged_df["pub_time"] = merged_df["pub_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save if output file specified
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Generated merged file: {output_file}")
        print(f"Total rows: {len(merged_df)}")

def save_to_db(csv_file, table_name, mode="append"):
    engine = create_engine(
        "mysql+pymysql://root:root@localhost:3306/stock"
    )

    df = pd.read_csv(csv_file)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=mode,
        index=False
    )
    print(f"Saved {len(df)} rows to table {table_name}")



alpha_news = "../data/alpha_nvda_news.csv"
finnhub_news = "../data/finnhub_nvda_news.csv"
massive_news = "../data/massive_nvda_news.csv"
marketaux_news = "../data/marketaux_nvda_news.csv"
merged_news = "../data/merged_nvda_news.csv"

yf_aapl_market_data = "../data/yf_aapl_market_data.csv"
yf_amzn_market_data = "../data/yf_amzn_market_data.csv"
yf_googl_market_data = "../data/yf_googl_market_data.csv"
yf_ibm_market_data = "../data/yf_ibm_market_data.csv"
yf_msft_market_data = "../data/yf_msft_market_data.csv"
yf_nvda_market_data = "../data/yf_nvda_market_data.csv"

# chop_by_period("../data/finnhub_nvda_news.csv", "2023-05-01","2026-04-30")

# print(f"Processing file: {alpha_news}")
# find_missing_dates(alpha_news, "2023-05-01","2026-04-30")
# print(f"\nProcessing file: {finnhub_news}")
# find_missing_dates(finnhub_news, "2023-05-01","2026-04-30")
# print(f"\nProcessing file: {massive_news}")
# find_missing_dates(massive_news, "2023-05-01","2026-04-30")

# files = [alpha_news, massive_news, marketaux_news]
# merge_news_csv(files, merged_news)
# find_missing_dates(merged_news, "2023-05-01","2026-04-30")

# add_column(yf_msft_market_data, "ticker", "MSFT")

save_to_db(yf_nvda_market_data, "tbl_stock_market_data", "append")
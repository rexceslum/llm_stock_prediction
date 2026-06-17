import pandas as pd
import os
import re
# from thefuzz import fuzz
from tqdm import tqdm
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

def find_missing_dates(csv_file, date_col, start_date, end_date):
    data = pd.read_csv(csv_file, parse_dates=[date_col])
    existing_dates = set(data[date_col].dt.date)
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

def clean_news(csv_file):
    df = pd.read_csv(csv_file, parse_dates=["pub_time"])
    print(f"Initial total rows: {len(df)}")

    # Create date column to remove duplicates within same day only
    df["pub_date"] = df["pub_time"].dt.date

    # Clean up newlines, tabs, multi spaces, '', ""
    df["title"] = (df["title"].astype("string")
                   .str.replace(r"\s+", " ", regex=True)
                   .str.replace("’", "'", regex=False)
                   .str.replace("“", '"', regex=False)
                   .str.replace("”", '"', regex=False)
                   .str.strip()
    )
    df["summary"] = (df["summary"].astype("string")
                     .str.replace(r"\s+", " ", regex=True)
                     .str.replace("’", "'", regex=False)
                     .str.replace("“", '"', regex=False)
                     .str.replace("”", '"', regex=False)
                     .str.strip())

    # Remove non-English news
    english_pattern = re.compile(
        r"""^[a-zA-Z0-9\s
        .,!?;:'"’“”()\[\]{}
        \-–—_/\\
        $%&+*=#@
        ]+$""",
        re.VERBOSE,
    )

    title_is_english = df["title"].astype("string").str.fullmatch(english_pattern, na=False)

    summary_is_english_or_missing = (df["summary"].isna() | df["summary"].astype("string")
                                     .str.fullmatch(english_pattern, na=False)
    )

    df_cleaned = df[title_is_english & summary_is_english_or_missing].copy()

    # Remove duplicates and drop temp columns
    df_cleaned["title_lower"] = df_cleaned["title"].str.lower()
    df_cleaned = df_cleaned.drop_duplicates(subset=["pub_date", "title_lower"], keep="first")
    df_cleaned = df_cleaned.drop(columns=["pub_date", "title_lower", "sentiment_score"])
    df_cleaned["news_text"] = df_cleaned["title"].str.cat(df_cleaned["summary"].fillna(""), sep=". ").str.rstrip('.')

    # Append prefix on filename and save as new csv
    directory, filename = os.path.split(csv_file)
    new_path = os.path.join(directory, f"cleaned_{filename}")
    df_cleaned.to_csv(new_path, index=False)
    print("Final total rows: ", len(df_cleaned))
    print("Saved to CSV")

def group_news_by_date(input, output):
    df = pd.read_csv(input, parse_dates=["pub_time"])

    df["Date"] = df["pub_time"].dt.date
    excluded_columns = {
        "pub_time",
        "title",
        "summary",
        "news_text",
        "finbert_label",
        "sentiment_score",
    }
    numerical_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    aggregated_df = (
        df.groupby(["Date", "ticker"], as_index=False)
        .agg(
            **{
                column: (column, "mean")
                for column in numerical_columns
            },
            news_count=("ticker", "size"),
        )
        .sort_values(["Date", "ticker"])
        .reset_index(drop=True)
    )
    aggregated_df.to_csv(output, index=False)
    print("Total rows: ", len(aggregated_df))
    print("Saved to CSV")


# def remove_fuzzy_duplicate(csv_file):
#     df = pd.read_csv(csv_file, parse_dates=["pub_time"])
#     df["title_lower"] = df["title"].str.lower()
#
#     # Fuzzy match to find and filter similar strings (threshold set to 90/100)
#     unique_texts = df["title_lower"].unique()
#     rows_to_keep = []
#     for text in tqdm(
#         unique_texts,
#         total=len(unique_texts),
#         desc="Removing similar titles",
#         unit="title",
#     ):
#         if not rows_to_keep:
#             rows_to_keep.append(text)
#         else:
#             # Check against already kept texts
#             is_similar = False
#             for kept_text in rows_to_keep:
#                 if fuzz.ratio(text, kept_text) >= 90:
#                     is_similar = True
#                     break
#             if not is_similar:
#                 rows_to_keep.append(text)
#
#     df_cleaned = df[df["title_lower"].isin(rows_to_keep)]
#     return df_cleaned

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



ticker = "nvda"
alpha_news = f"../data/alpha_{ticker}_news.csv"
finnhub_news = f"../data/finnhub_{ticker}_news.csv"
massive_news = f"../data/massive_{ticker}_news.csv"
marketaux_news = f"../data/marketaux_{ticker}_news.csv"
eodhd_news = f"../data/eodhd_{ticker}_news.csv"
merged_news = f"../data/merged_{ticker}_news.csv"
clean_merged_news = f"../data/cleaned_merged_{ticker}_news.csv"
finbert_news = f"../data/finbert_{ticker}_news_sentiment.csv"
llm_news = f"../data/llm_{ticker}_news_sentiment.csv"
final_news = f"../data/final_{ticker}_news_sentiment.csv"

yf_aapl_market_data = "../data/yf_aapl_market_data.csv"
yf_amzn_market_data = "../data/yf_amzn_market_data.csv"
yf_googl_market_data = "../data/yf_googl_market_data.csv"
yf_ibm_market_data = "../data/yf_ibm_market_data.csv"
yf_msft_market_data = "../data/yf_msft_market_data.csv"
yf_nvda_market_data = "../data/yf_nvda_market_data.csv"

# chop_by_period("../data/finnhub_nvda_news.csv", "2023-05-01","2026-04-30")

# print(f"Processing file: {alpha_news}")
# find_missing_dates(alpha_news, "pub_time", "2023-05-01","2026-04-30")
# print(f"\nProcessing file: {finnhub_news}")
# find_missing_dates(finnhub_news, "pub_time", "2023-05-01","2026-04-30")
# print(f"\nProcessing file: {massive_news}")
# find_missing_dates(massive_news, "pub_time", "2023-05-01","2026-04-30")

# files = [alpha_news, massive_news, marketaux_news, eodhd_news]
# merge_news_csv(files, merged_news)
# find_missing_dates(merged_news, "pub_time", "2023-05-01","2026-04-30")

# clean_news(merged_news)
# find_missing_dates(clean_merged_news, "pub_time", "2023-05-01","2026-04-30")

# add_column(yf_msft_market_data, "ticker", "MSFT")

# save_to_db(yf_nvda_market_data, "tbl_stock_market_data", "append")

# group_news_by_date(finbert_news, final_news)
# find_missing_dates(final_news, "Date", "2023-05-01","2026-04-30")

# df = pd.read_csv(clean_merged_news)
# print(df["ticker"].unique())

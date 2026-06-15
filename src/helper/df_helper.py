import pandas as pd
import numpy as np

def eda(df):
    print("="*30)
    print("Data Snippet")
    print("="*30)
    print(df)

    # Check data types, non-null counts, and memory usage
    print("\n"+"=" * 30)
    print("Data Info")
    print("=" * 30)
    print(df.info())

    # Get total missing values per column
    print("\n"+"=" * 30)
    print("Missing Data Count")
    print("=" * 30)
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("No missing data")


    # Get summary statistics for numerical columns
    print("\n"+"=" * 30)
    print("Summary For Numerical Columns")
    print("=" * 30)
    pd.set_option('display.max_columns', None)  # Force pandas to show every single column
    print(df.describe())
    print("If mean is significantly larger than 50%, your data is right-skewed. \n"
          "If it is significantly smaller, it is left-skewed. If they are heavily skewed, use RobustScaler.")

    # Check percentage of outliers in dataset
    print("\n"+"=" * 30)
    print("Outlier Percentage")
    print("=" * 30)
    numeric_df = df.drop(columns=['Date', 'ticker']).copy()
    IQR(numeric_df)
    print("If outlier_percent > 1%, use RobustScaler. If it is 0%, use StandardScaler.")

    # Check skewness value
    print("\n" + "=" * 30)
    print("Skewness Value")
    print("=" * 30)
    for column in numeric_df.columns.tolist():
        print(f"Skewness value for {column}: {df[column].skew()}")
    print("A skewness value between -0.5 and 0.5 means the data is fairly symmetrical (Use StandardScaler). \n"
          "A skewness outside [-1, 1] means highly skewed data (Use RobustScaler).")

def IQR(df):
    for column in df.columns.tolist():
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Calculate outlier percentage
        outlier_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_percent = (outlier_count / len(df)) * 100

        print(f"Outlier Percentage for {column}: {outlier_percent:.2f}%")

def generate_stationary_features(df):
    # Intraday price relationships
    df["Open_Close_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]

    # Moving-average distances
    df["EMA_20_Dist"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
    df["SMA_50_Dist"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
    df["SMA_200_Dist"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"]

    # Volume features
    df["Log_Volume_Change"] = np.log1p(df["Volume"]).diff()
    df["Relative_Volume_20"] = (df["Volume"] / df["Volume"].rolling(20).mean())

    # Volatility
    df["ATR_14_Pct"] = df["ATR_14"] / df["Close"]

    # OBV
    df["OBV_Change_Normalized"] = (df["OBV"].diff() / df["Volume"].rolling(20).mean())

    # MACD
    df["MACD_Pct"] = (df["MACD_12_26_9"] / df["Close"])
    df["MACD_Hist_Pct"] = (df["MACDh_12_26_9"] / df["Close"])

    # Remove NaN and infinite values created by transformations
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
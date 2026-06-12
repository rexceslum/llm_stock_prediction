import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import os
from datetime import date

# Set up date range and ticker
end_date = date(2026, 4, 30)
buffer_end_date = date(2026, 5, 2)
start_date = date(2023, 5, 1)
buffer_start_date = date(2022, 5, 1)
ticker_symbol = "MSFT"

# Option A: Retrieve by start end date
data = yf.download(ticker_symbol, start=buffer_start_date, end=buffer_end_date, auto_adjust=True)

# Option B: Retrieve by period length
# data = yf.download(ticker_symbol, period="3y")

# Option C: Using yf.Ticker
# ticker = yf.Ticker(ticker_symbol)
# data = ticker.history(period="3y")

# If columns are multi-level (tuples), keep only the metric name (e.g., 'Close')
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Force Pandas to show all data
pd.set_option("display.max_rows", None)  # None removes the row limit
pd.set_option("display.max_columns", None)  # None removes the column limit
pd.set_option("display.width", 1000)  # Prevents lines from wrapping
print(data)

close_prices = data["Close"]
high_prices = data["High"]
low_prices = data["Low"]
volume_data = data["Volume"]

# Calculate Technical Indicators
data["Daily_Change_USD"] = close_prices.diff()
data["Daily_Return_Pct"] = close_prices.pct_change()
data["EMA_20"] = ta.ema(close_prices, length=20)
data["SMA_50"] = ta.sma(close_prices, length=50)
data["SMA_200"] = ta.sma(close_prices, length=200)
data["RSI_14"] = ta.rsi(close_prices, length=14)
macd_df = ta.macd(close_prices, fast=12, slow=26, signal=9)
bbands_df = ta.bbands(close_prices, length=20, std=2)
data["ATR_14"] = ta.atr(high_prices, low_prices, close_prices, length=14)
data["OBV"] = ta.obv(close_prices, volume_data)
stoch_df = ta.stoch(high_prices, low_prices, close_prices, k=14, d=3, smooth_k=3)
data = pd.concat([data, macd_df, bbands_df, stoch_df], axis=1)
data["Target_Forward_Return"] = data["Daily_Return_Pct"].shift(-1)  # Target variable regression
data["Target_Direction"] = (data["Target_Forward_Return"] > 0).astype(int)  # Target variable classification

# Cuts off the 2022 buffer data, starting exactly on 2023-05-01
data = data.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

# Saves everything including dates, prices, and technical markers
output_filename = f"../data/yf_{ticker_symbol.lower()}_market_data.csv"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
data.to_csv(output_filename)

print(f"\nSUCCESS: Full data successfully saved to '{output_filename}'")
print(f"Total entries written: {len(data)} rows.")
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
from datetime import date, timedelta

# Set up date range and ticker
end_date = date(2026, 5, 1)
start_date = date(2023, 5, 1)
buffer_start_date = date(2022, 5, 1)
ticker_symbol = "NVDA"

# Option A: Retrieve by start end date
data = yf.download(ticker_symbol, start=buffer_start_date, end=end_date, auto_adjust=True)

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
data["EMA_20"] = ta.ema(close_prices, length=20)
data["SMA_50"] = ta.sma(close_prices, length=50)
data["SMA_200"] = ta.sma(close_prices, length=200)
data["RSI_14"] = ta.rsi(close_prices, length=14)
macd_df = ta.macd(close_prices, fast=12, slow=26, signal=9)
bbands_df = ta.bbands(close_prices, length=20, std=2)
data["ATR_14"] = ta.atr(high_prices, low_prices, close_prices, length=14)
data["OBV"] = ta.obv(close_prices, volume_data)
data = pd.concat([data, macd_df, bbands_df], axis=1)

# Cuts off the 2022 buffer data, starting exactly on 2023-05-01
data = data.loc[pd.Timestamp(start_date) :]

# Saves everything including dates, prices, and technical markers
output_filename = "../data/yf_nvda_market_data.csv"
data.to_csv(output_filename)

print(f"\nSUCCESS: Full data successfully saved to '{output_filename}'")
print(f"Total entries written: {len(data)} rows.")
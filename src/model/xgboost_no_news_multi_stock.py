import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import spearmanr

from src.database import market_data_repo
from src.helper import df_helper, plotting_helper

# Lock Python's hash seed
os.environ['PYTHONHASHSEED'] = '42'
# Lock Python's built-in random number generator
random.seed(42)
# Lock Numpy's random number generator (Crucial for Pandas/Sklearn)
np.random.seed(42)
# Lock TensorFlow'srandom number generator (Crucial for the LSTM)
tf.random.set_seed(42)

# Load the CSV
df = pd.read_csv("../../data/yf_merged_market_data.csv", index_col="Date", parse_dates=True)
# df = market_data_repo.retrieve_by_ticker("NVDA")
df_helper.eda(df)

df.sort_values(by=['ticker', 'Date'], inplace=True)

# Change target variable to this (Predicting the next 5 days of cumulative return):
# We MUST group by ticker so the rolling window doesn't bleed AAPL into NVDA
days_forward = 5
df["Target_Forward_Return"] = df.groupby('ticker')["Daily_Return_Pct"].transform(
    lambda x: x.rolling(window=days_forward).sum().shift(-days_forward)
)

# Drop the final few rows because its Target columns are NaN (unknown future)
# We can't use it for training, but we keep it aside if we want to predict tomorrow live
df_clean = df.dropna(subset=["Target_Forward_Return"]).copy()

# Before splitting into X_raw, make 'Date' the index (if it isn't already)
# df_clean.set_index('Date', inplace=True)

# Apply one-hot encoding to the 'city' column
# df_clean = pd.get_dummies(df_clean, columns=['ticker'], dtype=int)

# Feature engineering to generate stationary features from non-stationary features
df_clean = df_helper.generate_stationary_features_multi(df_clean)

# One-hot encode the ticker to convert text into numeric labels
df_clean = pd.get_dummies(df_clean, columns=['ticker'], dtype=int)

# Separate Features (X) and Targets (y)
# We use Target_Forward_Return for a regression task
feature_cols = [
    # "ticker",
    "Daily_Return_Pct",         # OHLC indicators
    "Open_Close_Return",
    "High_Low_Range",
    "EMA_20_Dist",              # Trend indicators
    # "SMA_50_Dist",
    # "SMA_200_Dist",
    # "MACD_Pct",
    "MACD_Hist_Pct",
    "RSI_14",                   # Momentum indicators
    # "STOCHk_14_3_3",
    # "STOCHd_14_3_3",
    "ATR_14_Pct",               # Volatility indicators
    "BBP_20_2.0",
    # "BBB_20_2.0",
    "OBV_Change_Normalized",    # Volume indicators
    "Log_Volume_Change",
    # "Relative_Volume_20",
]
# Add one-hot encoded ticker columns to our feature list
ticker_cols = [col for col in df_clean.columns if col.startswith('ticker_')]
feature_cols.extend(ticker_cols)

df_clean.sort_index(inplace=True) # Sort chronologically for the global split

# Global Temporal Train/Test Split
# We split by unique dates to ensure the test set is purely in the future
unique_dates = df_clean.index.unique()
# split_idx = int(len(unique_dates) * 0.9)
# split_date = unique_dates[split_idx]
#
# train_df = df_clean[df_clean.index < split_date].copy()
# test_df = df_clean[df_clean.index >= split_date].copy()
# print(f"\nTrain date range: {train_df.index.min()} to {train_df.index.max()}")
# print(f"Test date range : {test_df.index.min()} to {test_df.index.max()}")

train_idx = int(len(unique_dates) * 0.8)
val_idx = int(len(unique_dates) * 0.9)
train_end_date = unique_dates[train_idx]
val_end_date = unique_dates[val_idx]
train_df = df_clean[df_clean.index < train_end_date].copy()
val_df = df_clean[(df_clean.index >= train_end_date) & (df_clean.index < val_end_date)].copy()
test_df = df_clean[df_clean.index >= val_end_date].copy()

print(f"\nTrain date range: {train_df.index.min()} to {train_df.index.max()}")
print(f"Val date range  : {val_df.index.min()} to {val_df.index.max()}")
print(f"Test date range : {test_df.index.min()} to {test_df.index.max()}")

print("\nTrain rows by ticker:")
print(train_df[ticker_cols].sum())
print("\nTest rows by ticker:")
print(test_df[ticker_cols].sum())

# Fit the Scaler ONLY on the Training Data (No Lookahead Bias!)
# scaler = StandardScaler()
scaler = RobustScaler()
# Scale only the continuous features, not the one-hot encoded tickers or targets
continuous_features = [col for col in feature_cols if col not in ticker_cols]

train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
val_df[continuous_features] = scaler.transform(val_df[continuous_features])
test_df[continuous_features] = scaler.transform(test_df[continuous_features])

X_train_2D = train_df[feature_cols].values
X_val_2D = val_df[feature_cols].values
X_test_2D = test_df[feature_cols].values
y_train_final = train_df["Target_Forward_Return"].values
y_val_final = val_df["Target_Forward_Return"].values
y_test_final = test_df["Target_Forward_Return"].values

# Verify Shapes
print("\n--- DATA PREPARATION COMPLETE ---")
print(f"X_train shape (Samples, Features): {X_train_2D.shape}")
print(f"X_val shape (Samples, Features): {X_val_2D.shape}")
print(f"X_test shape (Samples, Features):  {X_test_2D.shape}")
print(f"y_train shape (Total Targets): {y_train_final.shape}")
print(f"y_val shape (Total Targets): {y_val_final.shape}")
print(f"y_test shape (Total Targets): {y_test_final.shape}")

print("\n=== PHASE 1: Training the XGBoost Meta-Model ===")

# Train the XGBoost model on the combined feature set
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,          # Increased to allow for the smaller learning rate
    learning_rate=0.01,        # Much slower, more cautious learning
    max_depth=3,                # Shallower trees to prevent memorizing complex, noisy patterns
    min_child_weight=15,        # Forces leaves to have a minimum number of samples
    subsample=0.3,              # Only uses 50% of the rows per tree
    colsample_bytree=0.3,       # Only uses 50% of the columns per tree
    # reg_alpha=1.0,              # L1 Regularization (shrinks less important feature weights to 0)
    # reg_lambda=1.0,             # L2 Regularization (prevents weights from getting too large)
    random_state=42,
    eval_metric=['rmse', 'mae'],
    early_stopping_rounds=30    # Increased patience since the learning rate is smaller
)

xgb_model.fit(X_train_2D, y_train_final,
              eval_set=[(X_train_2D, y_train_final), (X_val_2D, y_val_final)],
              verbose=5)

# Generate rmse and mae graph of model training
plotting_helper.plot_xgboost_train_graph(xgb_model.evals_result())
plotting_helper.plot_feature_importance(xgb_model, feature_cols, [], "xgboost", "N")

print("\n=== PHASE 2: Final Evaluation ===")

# Make predictions
y_pred = xgb_model.predict(X_test_2D)

# 1. Standard Regression Metrics
mae = mean_absolute_error(y_test_final, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
r2 = r2_score(y_test_final, y_pred)

# 2. Quant Finance Metrics (Accuracy & IC)
correct_direction = np.sign(y_test_final) == np.sign(y_pred)
directional_accuracy = np.mean(correct_direction) * 100
ic, p_value = spearmanr(y_pred, y_test_final)

# 3. Simulate a Trading Strategy for the Sharpe Ratio
# We use percentiles to find the model's highest relative confidence setups
upper_threshold = np.percentile(y_pred, 75)  # Top 25% most bullish predictions
lower_threshold = np.percentile(y_pred, 25)  # Bottom 25% most bearish predictions

# Signal: +1 if strong UP (and positive), -1 if strong DOWN (and negative), 0 (Cash)
trading_signals = np.where((y_pred > upper_threshold) & (y_pred > 0), 1,
                           np.where((y_pred < lower_threshold) & (y_pred < 0), -1, 0))

# Strategy Return: Signal * Actual Daily Return
strategy_returns = trading_signals * y_test_final

# Calculate how often the model actually took a trade vs holding cash
time_in_market = np.mean(trading_signals != 0) * 100

# Annualize the returns and volatility dynamically based on days_forward
periods_per_year = 252 / days_forward

# We assume a risk-free rate of 0% for this basic simulation
if np.std(strategy_returns) != 0:
    annualized_return = np.mean(strategy_returns) * periods_per_year
    annualized_volatility = np.std(strategy_returns) * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility
else:
    annualized_return = 0.0
    annualized_volatility = 0.0
    sharpe_ratio = 0.0

print("--- Standard Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (R2): {r2:.4f}")

print("\n--- Quant Trading Metrics ---")
print(f"Directional Accuracy (Hit Rate): {directional_accuracy:.2f}%")
print(f"Information Coefficient (IC): {ic:.4f} (p-value: {p_value:.4f})")

print("\n--- Simulated Portfolio Performance ---")
print(f"Dynamic Long Threshold : > {upper_threshold * 100:.2f}%")
print(f"Dynamic Short Threshold: < {lower_threshold * 100:.2f}%")
print(f"Time in Market: {time_in_market:.2f}%")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Print a few examples to see real vs predicted
print("\nSample Predictions vs Actual Returns:")
for i in range(5):
    actual_pct = y_test_final[i] * 100
    pred_pct = y_pred[i] * 100

    if trading_signals[i] > 0:
        signal = "LONG "
    elif trading_signals[i] < 0:
        signal = "SHORT"
    else:
        signal = "CASH "

    profit = strategy_returns[i] * 100
    print(
        f"Day {i + 1} | Signal: {signal} | Pred: {pred_pct:+.2f}% | Actual: {actual_pct:+.2f}% | Profit: {profit:+.2f}%")

print("\n=== PHASE 3: Plotting Results ===")

# Dates align with targets: from the beginning up to the 'split_index'
train_dates = train_df.index.unique()
val_dates = val_df.index.unique()
test_dates = test_df.index.unique()

# Create the plot
plt.figure(figsize=(18, 10))

# Define a distinct color map for up to 5 stocks (Standard Matplotlib Tableau Colors)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Loop through each one-hot column
for i, ticker_col in enumerate(ticker_cols):
    # Strip the prefix to get the clean stock name for the legend (e.g., 'ticker_NVDA' -> 'NVDA')
    stock_name = ticker_col.replace('ticker_', '')
    color = colors[i % len(colors)]

    # Filter the dataframe where this specific stock is '1'
    stock_train_df = train_df[train_df[ticker_col] == 1]
    stock_val_df = val_df[val_df[ticker_col] == 1]
    stock_test_df = test_df[test_df[ticker_col] == 1]

    X_train_stock = stock_train_df[feature_cols].values
    X_val_stock = stock_val_df[feature_cols].values
    X_test_stock = stock_test_df[feature_cols].values
    y_train_actual = stock_train_df["Target_Forward_Return"].values
    y_val_actual = stock_val_df["Target_Forward_Return"].values
    y_test_actual = stock_test_df["Target_Forward_Return"].values

    y_train_pred = xgb_model.predict(X_train_stock)
    y_val_pred = xgb_model.predict(X_val_stock)
    y_test_pred = xgb_model.predict(X_test_stock)

    # Fetch the ACTUAL Prices for the BASE day (Today)
    base_train_price = stock_train_df['Close'].values
    base_val_price = stock_val_df['Close'].values
    base_test_price = stock_test_df['Close'].values

    # Convert predicted and actual returns to future PRICES
    # Math: Future Price = Today's Actual Price * (1 + Return)
    y_train_actual_price = base_train_price * (1 + y_train_actual)
    y_val_actual_price = base_val_price * (1 + y_val_actual)
    y_test_actual_price = base_test_price * (1 + y_test_actual)

    y_train_pred_price = base_train_price * (1 + y_train_pred)
    y_val_pred_price = base_val_price * (1 + y_val_pred)
    y_test_pred_price = base_test_price * (1 + y_test_pred)

    # 7. Plot Training Data (Faint/Transparent)
    plt.plot(train_dates, y_train_actual_price, color=color, alpha=0.5)
    plt.plot(train_dates, y_train_pred_price, color=color, alpha=0.5, linestyle=':')

    # 7. Plot Validation Data (Faint/Transparent)
    plt.plot(val_dates, y_val_actual_price, color=color, alpha=0.5)
    plt.plot(val_dates, y_val_pred_price, color=color, alpha=0.5, linestyle=':')

    # 8. Plot Testing Data (Bold)
    plt.plot(test_dates, y_test_actual_price, color=color, alpha=0.9, label=f'{stock_name} Actual')
    plt.plot(test_dates, y_test_pred_price, color=color, alpha=0.9, linestyle='--', label=f'{stock_name} Predicted')

# Draw a vertical dashed line exactly where the testing period begins
first_prediction_date = test_dates[0]
plt.axvline(x=first_prediction_date, color='black', linestyle='-.', linewidth=2, label='Train/Test Split')

# Annotate Metrics (Ensure mae, rmse, r2 exist in your scope from Phase 2)
metrics_text = (
    f"Overall Test Metrics:\n"
    f"RMSE: {rmse:.4f}\n"
    f"MAE: {mae:.4f}\n"
    f"R2: {r2:.4f}\n"
    f"Hit Rate: {directional_accuracy:.2f}%\n"
    f"IC: {ic:.4f} (p-value: {p_value:.4f})\n"
    f"Annual Return: {annualized_return:.4f}\n"
    f"Annual Volatility: {annualized_volatility:.4f}\n"
    f"Sharpe Ratio: {sharpe_ratio:.4f}"
)

props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
plt.gca().text(0.02, 0.96, metrics_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=props)

# Format the Graph
plt.title(f'Multi-Stock Ground Truth vs Predicted {days_forward}-Day Forward', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel(f'Cumulative {days_forward}-Day Return', fontsize=12)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

# Move legend outside so it doesn't cover the data
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Show the plot
plt.savefig(f"../../output/xgboost_multi_prediction_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png")
plt.close()
print("Saved prediction result to PNG")
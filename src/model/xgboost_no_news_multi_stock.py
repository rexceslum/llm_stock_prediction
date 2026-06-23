import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler

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
# df = pd.read_csv("../../data/yf_nvda_market_data.csv", index_col="Date")
df = market_data_repo.retrieve_by_ticker("NVDA")
df_helper.eda(df)

# Change target variable to this (Predicting the next 5 days of cumulative return):
df["Target_Forward_Return"] = df["Daily_Return_Pct"].shift(-5).rolling(window=5).sum()

# Drop the final few rows because its Target columns are NaN (unknown future)
# We can't use it for training, but we keep it aside if we want to predict tomorrow live
df_clean = df.dropna(subset=["Target_Forward_Return"]).copy()

# Before splitting into X_raw, make 'Date' the index (if it isn't already)
df_clean.set_index('Date', inplace=True)

# Apply one-hot encoding to the 'city' column
# df_clean = pd.get_dummies(df_clean, columns=['ticker'], dtype=int)

# Feature engineering to generate stationary features from non-stationary features
df_clean = df_helper.generate_stationary_features(df_clean)

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
X_raw = df_clean[feature_cols].values
y_raw = df_clean["Target_Forward_Return"].values

# Temporal Train/Test Split (Never use random shuffle for time-series stock data!)
# We split chronologically so the test set is entirely in the future relative to the training set.
split_index = int(len(X_raw) * 0.9)  # 90% Train, 10% Test

X_train_raw, X_test_raw = X_raw[:split_index], X_raw[split_index:]
y_train_raw, y_test_raw = y_raw[:split_index], y_raw[split_index:]

# Fit the Scaler ONLY on the Training Data (No Lookahead Bias!)
# scaler = StandardScaler()
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

# Transform the Test Data using the rules learned from the Train Data
X_test_scaled = scaler.transform(X_test_raw)

def create_sequences(features, targets, lookback):
    X_seq, y_seq = [], []
    for i in range(len(features) - lookback):
        # Extract a slice of 30 consecutive trading days
        X_seq.append(features[i : (i + lookback)])
        # The target corresponds to the day right after the window ends
        y_seq.append(targets[i + lookback])
    return np.array(X_seq), np.array(y_seq)

# Let's assume your model looks back at the past 30 trading days to predict tomorrow
lookback_days = 10

# Create sequences separately to ensure train/test don't bleed into each other during windowing
X_train_3D, y_train_final = create_sequences(X_train_scaled, y_train_raw, lookback_days)
X_test_3D, y_test_final = create_sequences(X_test_scaled, y_test_raw, lookback_days)

# Verify Shapes
print("\n--- DATA PREPARATION COMPLETE FOR HYBRID MODEL ---")
print(f"X_train shape (Samples, Time Steps, Features): {X_train_3D.shape}")
print(f"X_test shape  (Samples, Time Steps, Features): {X_test_3D.shape}")
print(f"y_train shape (Total Targets): {y_train_final.shape}")
print(f"y_test shape (Total Targets): {y_test_final.shape}")

print("\n=== PHASE 1: Preparing Data for XGBoost ===")

# Extract 2D Tabular Data (Taking the last day of each sequence)
# This perfectly aligns XGBoost's features with the targets
X_train_2D = X_train_3D[:, -1, :]
X_test_2D = X_test_3D[:, -1, :]

print(f"XGBoost Train Matrix Shape: {X_train_2D.shape}")

print("\n=== PHASE 2: Training the XGBoost Meta-Model ===")

# Train the XGBoost model on the combined feature set
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.06,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric=['rmse', 'mae'],
    early_stopping_rounds=20
)

xgb_model.fit(X_train_2D, y_train_final,
              eval_set=[(X_train_2D, y_train_final), (X_test_2D, y_test_final)],
              verbose=5)

# Generate rmse and mae graph of model training
plotting_helper.plot_xgboost_train_graph(xgb_model.evals_result())

print("\n=== PHASE 4: Final Evaluation ===")

# Make predictions
y_pred = xgb_model.predict(X_test_2D)

# Regression Metrics
mae = mean_absolute_error(y_test_final, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
r2 = r2_score(y_test_final, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (R2): {r2:.4f}")

# Print a few examples to see real vs predicted
print("\nSample Predictions vs Actual Returns:")
for i in range(5):
    actual_pct = y_test_final[i] * 100
    pred_pct = y_pred[i] * 100
    print(f"Day {i+1}: Actual: {actual_pct:+.2f}%  |  Predicted: {pred_pct:+.2f}%")

print("\n=== PHASE 5: Plotting Results ===")

# 1. Generate predictions for the training data
y_train_pred = xgb_model.predict(X_train_2D)

# 2. Extract corresponding dates for the plotted sequences
# Convert index to datetime objects for clean x-axis formatting
dates = pd.to_datetime(df_clean.index)

# Train dates align with y_train_final: from 'lookback_days' up to the 'split_index'
train_dates = dates[lookback_days:split_index]

# Test dates align with y_test_final: from 'split_index + lookback_days' to the end
test_dates = dates[split_index + lookback_days:]

# 3. Create the plot
plt.figure(figsize=(16, 8))

# Plot Training Data (Actual vs Predicted)
plt.plot(train_dates, y_train_final, label='Train Actual', color='steelblue', alpha=0.7)
plt.plot(train_dates, y_train_pred, label='Train Predicted', color='orange', alpha=0.7, linestyle='--')

# Plot Testing Data (Actual vs Predicted)
plt.plot(test_dates, y_test_final, label='Test Actual', color='darkgreen', alpha=0.7)
plt.plot(test_dates, y_pred, label='Test Predicted', color='magenta', alpha=0.7, linestyle='--')

# Draw a vertical dashed line at the exact split point
plt.axvline(x=test_dates[0], color='red', linestyle=':', label='Train/Test Split')

# 4. Annotate Metrics
# Build the text box containing the evaluated metrics
metrics_text = (
    f"Test Data Metrics:\n"
    f"RMSE: {rmse:.4f}\n"
    f"MAE: {mae:.4f}\n"
    f"$R^2$: {r2:.4f}"
)

# Properties for the text box styling
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')

# Place text box in the top-left corner (axes coordinates: 0.02, 0.96)
plt.gca().text(0.02, 0.96, metrics_text, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', bbox=props)

# 5. Format the Graph
plt.title('NVDA Ground Truth vs Predicted 5-Day Forward Returns', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)

# Improve x-axis date formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

# Display legend and grid
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.savefig(f"../../output/xgboost_prediction_graph_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png")
plt.close()
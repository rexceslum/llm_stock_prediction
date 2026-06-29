import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
days_forward = 1
# df["Target_Forward_Return"] = df.groupby('ticker')["Daily_Return_Pct"].transform(
#     lambda x: x.rolling(window=days_forward).sum().shift(-days_forward)
# )

# Drop rows with NaN targets (unknown future)
df_clean = df.dropna(subset=["Target_Forward_Return"]).copy()

# Before splitting into X_raw, make 'Date' the index (if it isn't already)
# df_clean.set_index('Date', inplace=True)

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

# Let's assume your model looks back at the past 20 trading days to predict tomorrow
lookback_days = 20

# Global Temporal Train/Test Split
# We split by unique dates to ensure the test set is purely in the future
unique_dates = df_clean.index.unique()
split_idx = int(len(unique_dates) * 0.9)
split_date = unique_dates[split_idx]
# To fill up the gap between training and testing set due to the lookback windows, we add a warmup period for testing set
warmup_date = unique_dates[split_idx - lookback_days]

train_df = df_clean[df_clean.index < split_date].copy()
test_df = df_clean[df_clean.index >= warmup_date].copy()
print(f"\nTrain date range: {train_df.index.min()} to {train_df.index.max()}")
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
test_df[continuous_features] = scaler.transform(test_df[continuous_features])

def create_sequences_multi(df_subset, features_list, lookback, ticker_columns):
    X_seq, y_seq = [], []

    # Create LSTM sequences without crossing ticker boundaries.
    for ticker_col in ticker_columns:
        stock_group = df_subset[df_subset[ticker_col] == 1]
        stock_group = stock_group.sort_index()

        features = stock_group[features_list].values
        targets = stock_group["Target_Forward_Return"].values

        for i in range(len(features) - lookback):
            X_seq.append(features[i: (i + lookback)])
            y_seq.append(targets[i + lookback])

    return np.array(X_seq), np.array(y_seq)

# Create sequences separately to ensure train/test don't bleed into each other during windowing
X_train_3D, y_train_final = create_sequences_multi(train_df, feature_cols, lookback_days, ticker_cols)
X_test_3D, y_test_final = create_sequences_multi(test_df, feature_cols, lookback_days, ticker_cols)

# Verify Shapes
print("\n--- DATA PREPARATION COMPLETE ---")
print(f"X_train shape (Samples, Time Steps, Features): {X_train_3D.shape}")
print(f"X_test shape (Samples, Time Steps, Features):  {X_test_3D.shape}")
print(f"y_train shape (Total Targets): {y_train_final.shape}")
print(f"y_test shape (Total Targets): {y_test_final.shape}")

num_features = X_train_3D.shape[2]

print("\n=== PHASE 1: Training the LSTM Feature Extractor ===")

# Build the LSTM Architecture
lstm_model = Sequential([
    Input(shape=(lookback_days, num_features)),

    # Layer 1: Wider LSTM to capture initial complex features
    # return_sequences=True is REQUIRED to pass 3D data to the next LSTM
    LSTM(128, return_sequences=True),
    Dropout(0.2),  # Increased dropout to fight overfitting on noise
    BatchNormalization(),  # Normalizes activations, helps prevent the "flatline" mean prediction

    # Layer 2: Funnels the sequence down into a single vector
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),

    # Layer 3: Dense feature mapping
    Dense(32, activation='relu'),

    # Final Output
    Dense(1, activation='linear')
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae', RootMeanSquaredError(name='rmse')])

# Train with Early Stopping (Stops training if the test set starts getting worse)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(
    X_train_3D, y_train_final,
    validation_data=(X_test_3D, y_test_final),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Generate loss and mae graph of model training
plotting_helper.plot_lstm_train_graph(history)

print("\n=== PHASE 2: Final Evaluation ===")

# Make predictions
y_pred = lstm_model.predict(X_test_3D).flatten()

# Standard Regression Metrics
mae = mean_absolute_error(y_test_final, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
r2 = r2_score(y_test_final, y_pred)

# Quant Finance Metrics (Accuracy & IC)
correct_direction = np.sign(y_test_final) == np.sign(y_pred)
directional_accuracy = np.mean(correct_direction) * 100
ic, p_value = spearmanr(y_pred, y_test_final)

# Simulate a Trading Strategy for the Sharpe Ratio
# Signal: +1 if predicted UP, -1 if predicted DOWN
trading_signals = np.where(y_pred > 0, 1, -1)

# Strategy Return: Signal * Actual Daily Return
strategy_returns = trading_signals * y_test_final

# Annualize the returns and volatility (assuming 252 trading days in a year)
# We assume a risk-free rate of 0% for this basic simulation
if np.std(strategy_returns) != 0:
    annualized_return = np.mean(strategy_returns) * 252
    annualized_volatility = np.std(strategy_returns) * np.sqrt(252)
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
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Print a few examples to see real vs predicted
print("\nSample Predictions vs Actual Returns:")
for i in range(5):
    actual_pct = y_test_final[i] * 100
    pred_pct = y_pred[i] * 100
    print(f"Day {i+1}: Actual: {actual_pct:+.2f}%  |  Predicted: {pred_pct:+.2f}%")

print("\n=== PHASE 5: Plotting Results (All Stocks) ===")

plt.figure(figsize=(18, 10))

# Define a distinct color map for up to 5 stocks (Standard Matplotlib Tableau Colors)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 2. Loop through each one-hot column
for i, ticker_col in enumerate(ticker_cols):
    # Strip the prefix to get the clean stock name for the legend (e.g., 'ticker_NVDA' -> 'NVDA')
    stock_name = ticker_col.replace('ticker_', '')
    color = colors[i % len(colors)]

    # 3. Filter the dataframe where this specific stock is '1'
    stock_train_df = train_df[train_df[ticker_col] == 1]
    stock_test_df = test_df[test_df[ticker_col] == 1]

    # 4. Generate sequences specifically for this stock
    # Your one-hot create_sequences_multi will handle this perfectly
    X_train_stock, y_train_actual = create_sequences_multi(stock_train_df, feature_cols, lookback_days, ticker_cols)
    X_test_stock, y_test_actual = create_sequences_multi(stock_test_df, feature_cols, lookback_days, ticker_cols)

    # 5. Generate predictions
    y_train_pred = lstm_model.predict(X_train_stock, verbose=0).flatten()
    y_test_pred = lstm_model.predict(X_test_stock, verbose=0).flatten()

    # 6. Extract corresponding dates
    train_dates = pd.to_datetime(stock_train_df.index)[lookback_days:]
    test_dates = pd.to_datetime(stock_test_df.index)[lookback_days:]

    # 7. Extract exact dates for the BASE day (Today)
    # We shift backwards by days_forward (1) to get the day the prediction was made FROM
    base_train_dates = stock_train_df.index[lookback_days - days_forward: -days_forward]
    base_test_dates = stock_test_df.index[lookback_days - days_forward: -days_forward]

    # 8. Fetch the ACTUAL Prices
    # Actual Price on Target Day (Ground Truth)
    y_train_actual_price = stock_train_df['Close'].loc[train_dates].values
    y_test_actual_price = stock_test_df['Close'].loc[test_dates].values

    # Actual Price on Base Day (Used to calculate predicted price)
    base_train_price = stock_train_df['Close'].loc[base_train_dates].values
    base_test_price = stock_test_df['Close'].loc[base_test_dates].values

    # 9. Convert predicted returns to predicted PRICES
    # Math: Predicted Price = Today's Actual Price * (1 + Predicted Return)
    y_train_pred_price = base_train_price * (1 + y_train_pred)
    y_test_pred_price = base_test_price * (1 + y_test_pred)

    # 7. Plot Training Data (Faint/Transparent)
    plt.plot(train_dates, y_train_actual_price, color=color, alpha=0.5)
    plt.plot(train_dates, y_train_pred_price, color=color, alpha=0.5, linestyle=':')

    # 8. Plot Testing Data (Bold)
    plt.plot(test_dates, y_test_actual_price, color=color, alpha=0.9, label=f'{stock_name} Actual')
    plt.plot(test_dates, y_test_pred_price, color=color, alpha=0.9, linestyle='--', label=f'{stock_name} Predicted')

# Draw a vertical dashed line exactly where the testing period begins
global_test_dates = pd.to_datetime(test_df.index.unique()).sort_values()
first_prediction_date = global_test_dates[lookback_days]
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
plt.title(f'Multi-Stock Ground Truth vs Predicted {days_forward}-Day Forward Returns', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel(f'Cumulative {days_forward}-Day Return', fontsize=12)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

# Move legend outside so it doesn't cover the data
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

file_name = f"../../output/lstm_multi_prediction_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png"
plt.savefig(file_name)
plt.close()
print("Saved prediction result to PNG")
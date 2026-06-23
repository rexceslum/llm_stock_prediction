import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
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
df = pd.read_csv("../../data/yf_merged_market_data.csv", index_col="Date")
# df = market_data_repo.retrieve_by_ticker("NVDA")
df_helper.eda(df)

# Change target variable to this (Predicting the next 5 days of cumulative return):
# We MUST group by ticker so the rolling window doesn't bleed AAPL into NVDA
df.sort_values(by=['ticker', 'Date'], inplace=True)
df["Target_Forward_Return"] = df.groupby('ticker')["Daily_Return_Pct"].transform(
    lambda x: x.rolling(window=5).sum().shift(-5)
)

# Drop rows with NaN targets (unknown future)
df_clean = df.dropna(subset=["Target_Forward_Return"]).copy()

# Feature engineering to generate stationary features from non-stationary features
df_clean = df_helper.generate_stationary_features_multi(df_clean)

# Optional but recommended: One-hot encode the ticker so the LSTM knows which stock it's looking at
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
split_idx = int(len(unique_dates) * 0.9)
split_date = unique_dates[split_idx]

train_df = df_clean[df_clean.index < split_date].copy()
test_df = df_clean[df_clean.index >= split_date].copy()
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

# Let's assume your model looks back at the past 20 trading days to predict tomorrow
lookback_days = 20

# Create sequences separately to ensure train/test don't bleed into each other during windowing
X_train_3D, y_train_final = create_sequences_multi(train_df, feature_cols, lookback_days, ticker_cols)
X_test_3D, y_test_final = create_sequences_multi(test_df, feature_cols, lookback_days, ticker_cols)

# Verify Shapes
print("\n--- DATA PREPARATION COMPLETE ---")
print(f"X_train shape: {X_train_3D.shape}")
print(f"X_test shape:  {X_test_3D.shape}")
print(f"y_train shape (Total Targets): {y_train_final.shape}")
print(f"y_test shape (Total Targets): {y_test_final.shape}")

num_features = X_train_3D.shape[2]

print("\n=== PHASE 1: Training the LSTM Feature Extractor ===")

# Build the LSTM Architecture
lstm_model = Sequential([
    Input(shape=(lookback_days, num_features)),

    # Layer 1: Wider LSTM to capture initial complex features
    # return_sequences=True is REQUIRED to pass 3D data to the next LSTM
    LSTM(64, return_sequences=True),
    Dropout(0.2),  # Increased dropout to fight overfitting on noise
    BatchNormalization(),  # Normalizes activations, helps prevent the "flatline" mean prediction

    # Layer 2: Funnels the sequence down into a single vector
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),

    # Layer 3: Dense feature mapping
    Dense(16, activation='relu'),

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
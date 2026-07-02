import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def plot_lstm_train_graph(history):
    # Extract metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    mae = history.history['mae']
    val_mae = history.history['val_mae']

    # Generate the plot
    plt.figure(figsize=(12, 5))

    # Plot RMSE
    plt.subplot(1, 2, 1)
    plt.plot(rmse, label='Training RMSE', color='blue')
    plt.plot(val_rmse, label='Validation RMSE', color='red')
    plt.title('LSTM: Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(mae, label='Training MAE', color='blue')
    plt.plot(val_mae, label='Validation MAE', color='red')
    plt.title('LSTM: Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"../../output/lstm_train_graph_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png")
    plt.close()

def plot_xgboost_train_graph(results, model="xgboost"):
    # Extract metrics
    train_rmse = results['validation_0']["rmse"]
    val_rmse = results['validation_1']["rmse"]
    train_mae = results['validation_0']['mae']
    val_mae = results['validation_1']['mae']

    # Generate the plot
    plt.figure(figsize=(12, 5))

    # Plot RMSE
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training RMSE', color='blue')
    plt.plot(val_rmse, label='Validation RMSE', color='red')
    plt.title('XGBoost: Training and Validation RMSE')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(train_mae, label='Training MAE', color='blue')
    plt.plot(val_mae, label='Validation MAE', color='red')
    plt.title('XGBoost: Training and Validation MAE')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"../../output/{model}_train_graph_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png")
    plt.close()

def plot_feature_importance(xgb_model, features_cols, sentiment_cols, model, lstm_involved="Y"):
    # --- FEATURE IMPORTANCE PLOT ---
    print("\nGenerating Feature Importance Plot...")

    # 1. Extract the raw importance scores (Gain) from the trained XGBoost model
    importances = xgb_model.feature_importances_

    # 2. Reconstruct the feature names in the EXACT order they were concatenated
    if "Y" == lstm_involved:
        lstm_names = [f"LSTM_Memory_Node_{i + 1}" for i in range(16)]
    else:
        lstm_names = []
    tech_names = features_cols  # Your 10 indicators + 5 ticker hot-encodes
    news_names = sentiment_cols  # Your sentiment features

    all_feature_names = lstm_names + tech_names + news_names

    # Verify dimensions match
    if len(importances) != len(all_feature_names):
        print(f"WARNING: Name mismatch! {len(importances)} features vs {len(all_feature_names)} names.")
    else:
        # 3. Build a DataFrame and sort it for a clean horizontal bar chart
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        # 4. Plot the results
        plt.figure(figsize=(12, 10))
        # Highlight the news features in a different color so they stand out
        colors = ['orange' if feat in news_names else 'skyblue' for feat in importance_df['Feature']]

        plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
        plt.title('XGBoost Feature Importance (Which signals drive the trades?)', fontsize=16, fontweight='bold')
        plt.xlabel('Relative Importance (Gain)', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 5. Save the plot
        feat_file = f"../../output/{model}_feature_importance_{datetime.now().strftime('%Y%m%dT%H%M%S')}.png"
        plt.savefig(feat_file)
        plt.close()
        print(f"Saved Feature Importance plot to {feat_file}")
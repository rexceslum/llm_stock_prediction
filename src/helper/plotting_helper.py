import matplotlib.pyplot as plt
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
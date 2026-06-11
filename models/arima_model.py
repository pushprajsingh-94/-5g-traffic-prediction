import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(test_hours=200):
    print("Loading data...")
    df = load_data()
    series = df['traffic_mbps']

    train = series[:-test_hours]
    test = series[-test_hours:]

    print(f"Train: {len(train)} | Test: {test_hours}")
    print("Training ARIMA model... (thoda time lagega)")

    model = ARIMA(train, order=(2, 1, 2))
    fitted = model.fit()

    print("Generating predictions...")
    forecast = fitted.forecast(steps=test_hours)

    y_true = test.values
    y_pred = forecast.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\nARIMA Results:")
    print(f"MAE  : {mae:.2f} Mbps")
    print(f"RMSE : {rmse:.2f} Mbps")
    print(f"R2   : {r2:.4f}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'actual': y_true, 'arima_predicted': y_pred}).to_csv(
        'results/arima_predictions.csv', index=False)

    print("Predictions saved!")
    return {'model': 'ARIMA', 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'y_true': y_true, 'y_pred': y_pred}

if __name__ == "__main__":
    train_and_evaluate()
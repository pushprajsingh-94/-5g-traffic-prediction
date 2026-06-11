import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_data, get_features, train_test_split_ts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate():
    print("Loading data...")
    df = load_data()
    df = get_features(df)

    features = ['hour', 'day_of_week', 'month', 'is_weekend',
                'lag_1', 'lag_24', 'lag_168',
                'rolling_mean_6', 'rolling_mean_24']
    target = 'traffic_mbps'

    train, test = train_test_split_ts(df)
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print("Training Linear Regression...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_true = y_test.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\nLinear Regression Results:")
    print(f"MAE  : {mae:.2f} Mbps")
    print(f"RMSE : {rmse:.2f} Mbps")
    print(f"R2   : {r2:.4f}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'actual': y_true, 'lr_predicted': y_pred}).to_csv(
        'results/lr_predictions.csv', index=False)

    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(model, 'saved_models/lr_model.pkl')
    print("Model saved!")

    return {'model': 'Linear Regression', 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'y_true': y_true, 'y_pred': y_pred}

if __name__ == "__main__":
    train_and_evaluate()
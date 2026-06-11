import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path='data/5g_traffic.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


def get_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['lag_1'] = df['traffic_mbps'].shift(1)
    df['lag_24'] = df['traffic_mbps'].shift(24)
    df['lag_168'] = df['traffic_mbps'].shift(168)
    df['rolling_mean_6'] = df['traffic_mbps'].rolling(6).mean()
    df['rolling_mean_24'] = df['traffic_mbps'].rolling(24).mean()
    df.dropna(inplace=True)
    return df


def train_test_split_ts(df, test_ratio=0.2):
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_5g_traffic(days=365, seed=42):
    np.random.seed(seed)
    start = datetime(2023, 1, 1)
    periods = days * 24
    timestamps = [start + timedelta(hours=i) for i in range(periods)]

    traffic = []
    for ts in timestamps:
        hour = ts.hour
        weekday = ts.weekday()

        base = 500

        daily = (
            300 * np.exp(-0.5 * ((hour - 9) / 2) ** 2) +
            400 * np.exp(-0.5 * ((hour - 20) / 2) ** 2) +
            100 * np.exp(-0.5 * ((hour - 13) / 2) ** 2)
        )

        if weekday >= 5:
            daily *= 1.2

        noise = np.random.normal(0, 50)
        spike = np.random.choice([0, np.random.uniform(300, 800)], p=[0.99, 0.01])
        value = max(50, base + daily + noise + spike)
        traffic.append(round(value, 2))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'traffic_mbps': traffic,
        'hour': [t.hour for t in timestamps],
        'day_of_week': [t.strftime('%A') for t in timestamps],
        'is_weekend': [1 if t.weekday() >= 5 else 0 for t in timestamps],
        'month': [t.month for t in timestamps]
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/5g_traffic.csv', index=False)
    print(f"Dataset ready: {len(df)} records")
    print(f"Traffic range: {df['traffic_mbps'].min():.1f} - {df['traffic_mbps'].max():.1f} Mbps")
    return df

if __name__ == "__main__":
    df = generate_5g_traffic()
    print(df.head(10))
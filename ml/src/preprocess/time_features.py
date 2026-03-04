import pandas as pd


def extract_time_features(df):
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df

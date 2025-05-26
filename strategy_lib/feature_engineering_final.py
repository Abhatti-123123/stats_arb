import pandas as pd
from sklearn.preprocessing import LabelEncoder
from . import feature_engineering

def engineer_features(df: pd.DataFrame, drop_redundant=True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = feature_engineering.engineer_features(df)
    selected_features = [
        'trace_gap_01', 'trace_gap_12', 'trace_r1', 'trace_r2',
        'IV_slope_10_30', 'SPY_CALL_IV_20', 'SPY_PUT_IV_20',
        'Tech_IV_diff', 'PUT_CALL_skew_20', 'PUT_CALL_skew_10',
        'XLK_SPY_corr', 'XLK_return_abs'
    ]
    df = df[selected_features + ["regime"]]
    for col in selected_features:
        df[f"{col}_z"] = (df[col] - df[col].rolling(20).mean()) / df[col].rolling(20).std()
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_ma5"] = df[col].rolling(5).mean()

    df = df[[f"{col}_z" for col in selected_features] +
            [f"{col}_lag1" for col in selected_features] +
            [f"{col}_ma5" for col in selected_features] +
            ['regime']]
    # df["regime_label"] = LabelEncoder().fit_transform(df["regime"])
    df["regime_label"] = df["regime"].astype("category").cat.codes
    df = df.drop(columns=['regime'])
    return df

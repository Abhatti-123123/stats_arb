import pandas as pd
from sklearn.preprocessing import LabelEncoder
from . import feature_engineering

def engineer_features(df: pd.DataFrame, drop_redundant=True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = feature_engineering.engineer_features(df)
    selected_features = [ #'Tech_IV_diff',
        'trace_gap_01', 'trace_gap_12', 'trace_r1',
        'IV_slope_10_30', 'SPY_CALL_IV_20', 'SPY_PUT_IV_20'
    ]
    df = df[selected_features + ["regime"]]
    ma_features = ['trace_gap_01', 'trace_gap_12', 'trace_r1']
    for col in selected_features:
        df[f"{col}_z"] = (df[col] - df[col].rolling(20).mean()) / df[col].rolling(20).std()
        df[f"{col}_lag1"] = df[col].shift(1)
        if col in ma_features:
            df[f"{col}_ma5"] = df[col].rolling(5).mean()
    
    df = df[[f"{col}_z" for col in selected_features] +
            [f"{col}_lag1" for col in selected_features] +
            [f"{col}_ma5" for col in ma_features] +
            ['regime']]
    # df["regime_label"] = LabelEncoder().fit_transform(df["regime"])
    df["regime_label"] = df["regime"].astype("category").cat.codes
    df = df.drop(columns=['regime'])
    return df

#trend: IV_slope_10_30, SPY_CALL_IV_20, SPY_PUT_IV_20, XLK_return_abs
#mean: trace_gap_01, trace_gap_12, trace_r1,

# meta_model_pipeline.py

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- STAGE 1: Basic FeatureSet Construction ---
def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    INPUT: DataFrame with columns like:
        ['trace_r0', 'trace_r1', 'trace_r2', 'XLK_return', 'SPY_IV', ...]
    OUTPUT: Clean feature DataFrame and labels (regimes)
    """
    df = df.dropna()
    df['trace_gap_01'] = df['trace_r0'] - df['trace_r1']
    df['trace_gap_12'] = df['trace_r1'] - df['trace_r2']
    features = df[['trace_r0', 'trace_r1', 'trace_r2', 'trace_gap_01', 'trace_gap_12',
                   'XLK_return', 'SPY_IV', 'XLK_SPY_corr']]
    y = df['regime']
    return features, y

# --- STAGE 2: Model Training ---
def train_regime_classifier(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Trains a basic random forest classifier. Future version can add time-aware splits.
    """
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    return pipe

# --- STAGE 3: Inference Wrapper ---
def predict_next_regime(model: Pipeline, latest_features: pd.Series) -> str:
    X_latest = latest_features.values.reshape(1, -1)
    return model.predict(X_latest)[0]

# --- USAGE (placeholder until we build full training loop) ---
if __name__ == "__main__":
    # Placeholder: you will load real merged dataframe from engineered regime data
    df = pd.read_csv("meta_features_sample.csv")  # Should include 'regime' column
    X, y = build_feature_set(df)
    model = train_regime_classifier(X, y)

    # Predict regime for next window
    example = X.iloc[-1]  # use latest row
    print("Predicted next regime:", predict_next_regime(model, example))

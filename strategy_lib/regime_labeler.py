# regime_labeler.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_trace_features(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds gap features to rolling Johansen trace statistics.
    Input columns: ['r=0', 'r=1', 'r=2']
    Output includes: ['gap_01', 'gap_12']
    """
    df = trace_df.copy()
    df['gap_01'] = df['r=0'] - df['r=1']
    df['gap_12'] = df['r=1'] - df['r=2']
    return df

def label_trace_regimes(trace_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Labels regimes using KMeans clustering on trace features.
    Returns full DataFrame with trace values, gaps, cluster and regime labels.
    """
    df = compute_trace_features(trace_df.copy())
    feature_cols = ['r=0', 'r=1', 'r=2', 'gap_01', 'gap_12']
    X = StandardScaler().fit_transform(df[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X)

    # Interpret regimes based on structure of trace gaps
    cluster_means = df.groupby('cluster')[feature_cols].mean()
    gap_01_max = cluster_means['gap_01'].idxmax()
    gap_12_min = cluster_means['gap_12'].idxmin()

    cluster_to_regime = {}
    for cluster in cluster_means.index:
        if cluster == gap_01_max:
            cluster_to_regime[cluster] = 'mean-revert'
        elif cluster == gap_12_min:
            cluster_to_regime[cluster] = 'trend'
        else:
            cluster_to_regime[cluster] = 'neutral'

    df['regime'] = df['cluster'].map(cluster_to_regime)
    full_index = pd.date_range(trace_df.index.min(), trace_df.index.max(), freq='B')
    df = df.reindex(full_index).ffill()
    df.index.name = 'Date'
    return df

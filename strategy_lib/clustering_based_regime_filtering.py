import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_trace_features(trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds gap features to rolling Johansen trace statistics.
    Input columns: ['r0', 'r1', 'r2']
    """
    df = trace_df.copy()
    df["gap_01"] = df["r=0"] - df["r=1"]
    df["gap_12"] = df["r=1"] - df["r=2"]
    return df

def label_clusters_kmeans(trace_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Performs KMeans clustering and assigns regimes to clusters based on structure.
    """
    features = compute_trace_features(trace_df)
    feature_cols = ["r=0", "r=1", "r=2", "gap_01", "gap_12"]
    X = StandardScaler().fit_transform(features[feature_cols])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X)
    features["cluster"] = clusters

    # Compute cluster-wise means
    cluster_means = features.groupby("cluster")[feature_cols].mean()

    # Heuristic logic to assign regimes based on gap structure
    gap_01_max = cluster_means["gap_01"].idxmax()
    gap_12_min = cluster_means["gap_12"].idxmin()

    cluster_to_regime = {}
    for cluster in cluster_means.index:
        if cluster == gap_01_max:
            cluster_to_regime[cluster] = "mean-revert"
        elif cluster == gap_12_min:
            cluster_to_regime[cluster] = "trend"
        else:
            cluster_to_regime[cluster] = "neutral"

    features["regime"] = features["cluster"].map(cluster_to_regime)
    return features[["regime"]]

def merge_regime_with_spread_index(spread_index: pd.Index, regime_series: pd.Series) -> pd.Series:
    """
    Aligns regime series to spread test period via ffill.
    """
    aligned = regime_series.reindex(spread_index).ffill().fillna("neutral")
    return aligned

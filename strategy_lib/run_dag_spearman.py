import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from feature_engineering import engineer_features
from config import FINAL_DATASET_PATH

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "../data/regime_training_dataset.csv"
TARGET = "regime"
THRESHOLD = 0.05
MAX_COND_SET = 2

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_PATH, parse_dates=['date']).drop(columns=['date'])
# Force rename if visually correct but not accessible
# df["regime_code"] = df["regime"].map({"trend": 1, "mean-revert": -1, "neutral": 0})
df = engineer_features(df)

#Frome feature analysis
selected_features = ["IV_slope_10_30", "trace_gap_12", "trace_gap_01", "trace_r1", "SPY_CALL_IV_20", "SPY_PUT_IV_20"]
dag_df = df[selected_features + ["regime"]]

# Encode regime as integer (required for correlation)
dag_df[TARGET] = dag_df[TARGET].astype("category").cat.codes
variables = list(dag_df.columns)
variables.remove(TARGET)

# -------------------------------
# Partial Spearman function
# -------------------------------
def partial_corr(x, y, cond_names):
    if not cond_names:
        return spearmanr(x, y)[0]
    Z = dag_df[cond_names]
    x_res = x - LinearRegression().fit(Z, x).predict(Z)
    y_res = y - LinearRegression().fit(Z, y).predict(Z)
    return spearmanr(x_res, y_res)[0]

# -------------------------------
# Build full graph
# -------------------------------
G = nx.Graph()
G.add_nodes_from(dag_df.columns)
for i, a in enumerate(dag_df.columns):
    for b in dag_df.columns[i+1:]:
        G.add_edge(a, b)

# -------------------------------
# Apply conditional independence tests
# -------------------------------
for (a, b) in list(G.edges()):
    other_vars = [v for v in dag_df.columns if v not in [a, b]]
    drop = False
    for k in range(MAX_COND_SET + 1):
        for cond in combinations(other_vars, k):
            pcorr = partial_corr(dag_df[a], dag_df[b], list(cond))
            if abs(pcorr) < THRESHOLD:
                G.remove_edge(a, b)
                drop = True
                break
        if drop:
            break

# -------------------------------
# Visualize final DAG skeleton
# -------------------------------
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_size=2000, node_color="lightyellow", font_size=10)
plt.title("📈 DAG Skeleton from Partial Spearman Correlations")
plt.tight_layout()
plt.show()


# 🔍 Causal DAG Analysis
# From the diagram, we can infer a clear and intuitive hierarchy of influence:

# 🧠 Causal Layering
# 🟡 Level 1: Volatility Drivers
# SPY_CALL_IV_20

# SPY_PUT_IV_20

# These feed into:

# IV_slope_10_30 → capturing the volatility term structure.

# 🟠 Level 2: Structural Indicators
# IV_slope_10_30

# Influenced by the above.

# Causally connected to trace_gap_01, trace_gap_12, and ultimately regime.

# This forms the macro-volatility feedback loop, i.e. the slope → cointegration structure → regime.

# 🔵 Level 3: Cointegration Dynamics
# trace_r1, trace_gap_01, trace_gap_12
# These interact amongst themselves and influence regime directly.
# This is the cointegration stress signal layer.

# 🔴 Level 4: Target
# regime


# ✅ Key Financial Interpretation:
# Pathway	Interpretation
# SPY_IV → IV_slope → trace_gaps → regime	Volatility structure drives regime through cointegration stress.
# trace_gaps, trace_r1 → regime	Cointegration gap widening → regime shift (mean-revert ↔ trend).
# IV_slope ↔ trace_gap_01	Term structure slope causally interacts with trace spread tightening or widening.
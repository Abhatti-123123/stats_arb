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
selected_features = ["Tech df", "PUT_CALL_skew_30", "Tech_IV_diff", "PUT_CALL_skew_20"]
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


# 🧠 Interpretation of the DAG Split:
# Cluster	Features	Interpretation
# A	Tech_IV_diff → Tech df → regime	Trend momentum predictors — IV trends drive regime
# B	PUT_CALL_skew_30 → PUT_CALL_skew_20	Mean-reversion sentiment predictors — risk reversal structure
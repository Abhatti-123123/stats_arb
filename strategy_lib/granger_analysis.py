import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def run_granger_causality_tests(df, regime_col="regime", max_lag=3, regime_class="trend"):
    """
    Test each feature for Granger causality with respect to binary regime.
    Returns: DataFrame with p-values for each lag.
    """
    df = df.copy()

    # Convert regime to binary
    df["is_trend"] = (df[regime_col] == regime_class).astype(int)

    # Drop non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["is_trend"], errors="ignore")
    result_rows = []

    for col in numeric_df.columns:
        try:
            test_data = df[[col, "is_trend"]].dropna()
            gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag+1):
                p_val = gc_result[lag][0]['ssr_ftest'][1]
                result_rows.append({
                    "Feature": col,
                    "Lag": lag,
                    "P-Value": p_val
                })
        except Exception as e:
            print(f"⚠️ Skipping {col}: {e}")
    
    results_df = pd.DataFrame(result_rows)
    return results_df.pivot(index="Feature", columns="Lag", values="P-Value").sort_values(by=1)

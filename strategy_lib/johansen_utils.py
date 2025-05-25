
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from strategy_lib.regime import detect_regime_from_johansen
from strategy_lib.regime import detect_regime

def compute_rolling_johansen_trace(log_data, window_size=252, step_size=10):
    trace_stats = []
    for start in range(0, len(log_data) - window_size, step_size):
        window = log_data.iloc[start:start + window_size]
        try:
            johansen_result = coint_johansen(window, det_order=1, k_ar_diff=2)
            trace_stats.append(johansen_result.lr1)
        except Exception:
            trace_stats.append([np.nan] * len(log_data.columns))
    trace_df = pd.DataFrame(trace_stats, columns=[f"r={i}" for i in range(len(log_data.columns))])
    trace_df.index = log_data.index[window_size::step_size][:len(trace_df)]
    trace_df["gap_01"] = trace_df["r=0"] - trace_df["r=1"]
    trace_df["gap_12"] = trace_df["r=1"] - trace_df["r=2"]
    trace_df["avg"]    = trace_df[["r=0", "r=1", "r=2"]].mean(axis=1)
    return trace_df

def compute_regime_series(input_data):
    if isinstance(input_data, pd.DataFrame):
        return input_data.apply(
            lambda row: detect_regime_from_johansen(row['r=0'], row['r=1'], row['r=2']),
            axis=1
        )
    else:
        # Assume it's a spread series
        return detect_regime(input_data)



from statsmodels.tsa.stattools import adfuller
import numpy as np

def detect_regime(spread, adf_pval_thresh=0.05):
    pval = adfuller(spread.dropna())[1]
    if pval < adf_pval_thresh:
        return "mean-revert"
    else:
        return "trend"

def detect_regime_from_trace(trace_stat_r0, threshold=25):
    """
    trace_stat_r0: float, the Johansen trace statistic for H0: r <= 0
    threshold: minimum value required to assume cointegration is stable
    """
    if np.isnan(trace_stat_r0):
        return "unknown"
    elif trace_stat_r0 > threshold:
        return "mean-revert"
    else:
        return "trend"

def detect_regime_from_johansen(trace_r0, trace_r1, trace_r2):
    gap_01 = trace_r0 - trace_r1
    gap_12 = trace_r1 - trace_r2

    # Mean-reverting regime: strong first eigenvalue, fast decay
    if gap_01 > 10 and gap_12 > 5 and trace_r0 > 40:
        return "mean-revert"

    # Trending regime: all eigenvalues are strong and close
    elif gap_01 < 6 and gap_12 < 4 and trace_r2 > 15:
        return "trend"

    # Otherwise ambiguous
    else:
        return "neutral"


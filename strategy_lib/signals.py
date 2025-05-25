
import pandas as pd
import numpy as np
from collections import Counter

def trend_signal(spread, window=10):
    ma = spread.rolling(window).mean()
    return (ma.diff() > 0).astype(int) * 2 - 1

def mean_reversion_signal(spread, upper, lower, close_thresh):
    signal = pd.Series(0, index=spread.index)
    state = 0
    for t in spread.index:
        val = spread.at[t]
        if state == 0 and val < lower:
            state = 1
        elif state == 0 and val > upper:
            state = -1
        elif state == 1 and abs(val) < close_thresh:
            state = 0
        elif state == -1 and abs(val) < close_thresh:
            state = 0
        signal.at[t] = state
    return signal

# def hybrid_signal(spread, regime_series, upper, lower, close_thresh, trend_window=10):
#     """Switch between mean-reversion and trend-following signals based on regime label per timestamp."""
#     valid_regime = regime_series.copy()
#     valid_regime = valid_regime.loc[valid_regime.index <= spread.index[-1]]  # limit to test period end
#     valid_regime = valid_regime.reindex(spread.index, method="ffill")  # fill regime labels forward

#     trend_sig = trend_signal(spread, window=trend_window)
#     revert_sig = mean_reversion_signal(spread, upper, lower, close_thresh)

#     hybrid = pd.Series(0, index=spread.index)

#     # Smart masking
#     trend_mask = regime_series == "trend"
#     revert_mask = regime_series == "mean-revert"

#     hybrid[trend_mask] = trend_sig[trend_mask]
#     hybrid[revert_mask] = revert_sig[revert_mask]

#     return hybrid
def infer_regime_from_series(regime_series: pd.Series, window_days: int = 126, neutral_threshold: float = 0.8) -> str:
    """
    Infers the dominant regime from the last `window_days` of regime_series.

    - If one regime has ≥70% of the recent window, it's chosen.
    - Otherwise:
        - If mix includes trend & mean-revert → fallback to "trend"
        - If only neutral dominates but <70% → fallback to "trend" (i.e., avoid indecision)

    Args:
        regime_series: pd.Series with datetime index and string regime labels
        window_days: how many recent days to consider
        neutral_threshold: minimum % of 'neutral' needed to call it neutral

    Returns:
        str: inferred regime ("trend", "mean-revert", or "neutral")
    """
    if not isinstance(regime_series, pd.Series) or regime_series.empty:
        return "neutral"

    recent = regime_series.iloc[-window_days:]
    counts = Counter(recent)
    total = sum(counts.values())

    if not counts:
        return "neutral"

    most_common, freq = counts.most_common(1)[0]
    pct = freq / total

    if most_common == "neutral" and pct < neutral_threshold:
        # Not dominant enough to trust
        # Fallback: if other regimes present, prefer trend (safer)
        if any(reg in counts for reg in ("trend", "mean-revert")):
            return "trend"
        else:
            return "neutral"
    return most_common

def hybrid_signal(spread, regime_series, upper, lower, close_thresh, trend_window=10):
    """
    Assign fixed regime to all test dates based on most recent regime detected in training.
    """
    last_known_regime = ""
    if isinstance(regime_series, pd.DataFrame):
        regime_series = regime_series['regime']
    
    if isinstance(regime_series, str):
        last_known_regime = regime_series
    elif isinstance(regime_series, pd.Series) and not regime_series.empty:
        last_known_regime = infer_regime_from_series(regime_series)
    else:
        last_known_regime = "neutral"

    if last_known_regime == "trend":
        signal = trend_signal(spread, window=trend_window)
    elif last_known_regime == "mean-revert":
        signal = mean_reversion_signal(spread, upper, lower, close_thresh)
    else:
        signal = pd.Series(0, index=spread.index)

    # Always ensure the signal index matches spread
    signal.name = "Signal"
    signal.index = spread.index
    return signal, last_known_regime




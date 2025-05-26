
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

def mean_reversion_signal_zscore(
    spread: pd.Series,
    z_entry: float = 1.5,
    z_exit: float = 0.5,
    window: int = 20,
    min_std: float = 1e-4
) -> pd.Series:
    """
    Generates a mean-reverting signal using z-score bands on the spread.

    Parameters:
    - spread : pd.Series, assumed to be stationary (e.g. from cointegration)
    - z_entry : float, entry threshold (abs(z) > z_entry triggers entry)
    - z_exit : float, exit threshold (abs(z) < z_exit triggers exit)
    - window : int, rolling window for mean and std
    - min_std : float, small constant to prevent division by zero

    Returns:
    - signal : pd.Series of +1 (long), -1 (short), or 0 (flat)
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std().clip(lower=min_std)
    zscore = (spread - rolling_mean) / rolling_std

    signal = pd.Series(0, index=spread.index)
    state = 0  # 0 = flat, 1 = long, -1 = short

    for t in spread.index:
        if rolling_std[t] < min_std or abs(zscore[t]) < 0.2:
            continue  # ignore weak or unstable signals
        vol = spread.pct_change().rolling(20).std()
        if vol[t] > vol.quantile(0.9):
            continue  # skip overactive periods
        z = zscore[t]
        if state == 0:
            if z < -z_entry:
                state = 1
            elif z > z_entry:
                state = -1
        elif state == 1 and z > -z_exit:
            state = 0
        elif state == -1 and z < z_exit:
            state = 0
        signal[t] = state

    return signal

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

from collections import Counter

def blended_regime_signal_bayesian(sig_trend, sig_revert, regime_probas, regime_prior=None, threshold=0.4):
    """
    Bayesian regime selector using posterior over entire test window.
    Assumes posterior is fixed over test window (like model confidence).

    - Returns sig_trend or sig_revert if confident.
    - Else returns weighted average signal (constant).
    """
    avg_probs = regime_probas.mean()

    if regime_prior is None:
        regime_prior = {'trend': 1/3, 'mean-revert': 1/3, 'neutral': 1/3}

    posterior = {}
    for regime in ['trend', 'mean-revert', 'neutral']:
        prior = regime_prior.get(regime, 0)
        likelihood = avg_probs.get(regime, 0)
        posterior[regime] = prior * likelihood

    total = sum(posterior.values())
    posterior = {k: v / total if total > 0 else 0 for k, v in posterior.items()}
    print("Avg probs:", avg_probs.to_dict())
    print("Posterior:", posterior)

    if posterior['trend'] > threshold:
        print("Chosen regime:", 'trend')
        return sig_trend.copy(), 'trend'
    elif posterior['mean-revert'] > threshold:
        print("Chosen regime:", 'mean-revert')
        return sig_revert.copy(), 'mean-revert'
    else:
        # FIX: Create blended signal as SERIES, not scalar mix
        blended = posterior['trend'] * sig_trend + posterior['mean-revert'] * sig_revert
        print("Chosen regime:", 'neutral')
        return blended, 'neutral'
    

def predictive_regime_signal(sig_trend, sig_revert, regime_probas, threshold=0.45):
    """
    Smarter regime signal logic:
    - If posterior max > threshold → pick that regime
    - If not → use confidence-weighted blend of trend/revert
    """
    avg_probs = regime_probas.ewm(span=10).mean().iloc[-1]
    chosen = avg_probs.idxmax()
    confidence = avg_probs[chosen]

    # CASE 1: Confident classification
    if chosen == 'trend' and confidence > threshold:
        return sig_trend.copy(), 'trend'
    elif chosen == 'mean-revert' and confidence > threshold:
        return sig_revert.copy(), 'mean-revert'
    elif chosen == 'neutral' and confidence > threshold:
        return pd.Series(0, index=sig_trend.index), 'neutral'

    # CASE 2: Soft blend (non-confident regime)
    trend_weight = avg_probs.get('trend', 0.0)
    revert_weight = avg_probs.get('mean-revert', 0.0)
    total_weight = trend_weight + revert_weight
    if total_weight == 0:
        return pd.Series(0, index=sig_trend.index), 'neutral'

    trend_weight /= total_weight
    revert_weight /= total_weight

    blended = trend_weight * sig_trend + revert_weight * sig_revert
    return blended, 'blend'



import numpy as np
import pandas as pd

def backtest_alpha_scaled_basket(df, beta, signal, fee_per_trade=0.001):
    """
    df     : DataFrame with cointegrated assets (e.g., ['SPY','QQQ','XLK'])
    beta   : vecm_res.beta[:, 0] → cointegration vector (np.array)
    signal : pd.Series of +1 / -1 / 0 for long/short/flat on the spread basket
    """
    basket_ret = df.pct_change().dropna() @ beta

    # Align signal
    signal = signal.reindex(df.index).fillna(0).astype(int)
    pos = signal.shift(1).reindex(basket_ret.index).fillna(0).astype(int)
    strat_ret = pos * basket_ret

    # Apply fees
    abs_sum = np.sum(np.abs(beta))
    per_trade_cost = fee_per_trade * abs_sum  # like slippage × turnover per rebalance
    entries = (pos.shift(1) != pos) & (pos != 0)
    strat_ret.loc[entries] -= per_trade_cost

    # Metrics
    equity = (1 + strat_ret).cumprod()
    days = len(basket_ret)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (252 / days) - 1
    ann_vol = strat_ret.std(ddof=1) * np.sqrt(252)
    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() != 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    nz = strat_ret[strat_ret != 0]
    hit_rate = (nz > 0).sum() / len(nz) if len(nz) else np.nan

    # Hold durations
    hold_durations = []
    current_pos = 0
    entry_date = None

    for date, new_pos in pos.items():
        if current_pos == 0 and new_pos != 0:
            entry_date = date  # Start of new position
        elif current_pos != 0 and new_pos != current_pos:
            if entry_date is not None:
                hold_durations.append((date - entry_date).days)
            entry_date = date if new_pos != 0 else None  # If flipped, this is a new entry

        current_pos = new_pos

    # Handle open position at end of backtest
    if current_pos != 0 and entry_date is not None:
        hold_durations.append((pos.index[-1] - entry_date).days)

    avg_hold = np.mean(hold_durations) if hold_durations else np.nan

    return {
        'Total Return': total,
        'CAGR': cagr,
        'Ann Vol': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Trades': int(entries.sum()),
        'Hit Rate': hit_rate,
        'Avg Hold Days': avg_hold,
        'Data Points': days
    }


def backtest_alpha_scaled_basket_soft(df, beta, signal, fee_per_trade=0.001, trade_thresh=0.05):
    """
    Handles float-weighted signals with reduced turnover cost overestimation.
    Uses smoothed signal and penalizes only significant changes.
    """
    df = df.copy()
    basket_ret = df.pct_change().dropna() @ beta

    signal = signal.reindex(df.index).fillna(0)
    pos = signal.shift(1).reindex(basket_ret.index).fillna(0)
    strat_ret = pos * basket_ret

    # Penalize only meaningful position change
    delta = pos.diff().abs().fillna(0)
    meaningful_change = delta > trade_thresh
    turnover_cost = meaningful_change.astype(float) * fee_per_trade * np.sum(np.abs(beta))
    strat_ret -= turnover_cost

    # Core metrics
    equity = (1 + strat_ret).cumprod()
    days = len(basket_ret)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1]**(252 / days) - 1
    ann_vol = strat_ret.std(ddof=1) * np.sqrt(252)
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()

    # Hit rate only on active positions
    valid = strat_ret[pos != 0]
    hit_rate = (valid > 0).mean() if not valid.empty else np.nan

    return {
        'Total Return': total,
        'CAGR': cagr,
        'Ann Vol': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Trades': int(meaningful_change.sum()),
        'Hit Rate': hit_rate,
        'Avg Hold Days': np.nan,
        'Data Points': days
    }

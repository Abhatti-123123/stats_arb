
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def plot_spread_diagnostics(spread, equity, signal, title="Spread Diagnostics", wipeout_threshold=0.0):
    spread = spread.copy()
    signal = signal.copy()
    equity = equity.copy()

    spread.index = pd.to_datetime(spread.index)
    signal.index = pd.to_datetime(signal.index)
    equity.index = pd.to_datetime(equity.index)

    # Focus range: 2018-03 to 2018-07 (you can customize)
    focus_start = "2018-03-01"
    focus_end   = "2018-07-01"

    spread_win = spread.loc[focus_start:focus_end]
    signal_win = signal.loc[focus_start:focus_end]
    equity_win = equity.loc[focus_start:focus_end]

    # Wipeout detection
    wipe_dates = equity_win[equity_win <= wipeout_threshold].index

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot spread
    ax1.plot(spread_win, label="Spread", color="black")
    ax1.set_ylabel("Spread")
    ax1.set_title(title)

    # Mark long/short entries
    long_days  = signal_win[signal_win == 1].index
    short_days = signal_win[signal_win == -1].index
    ax1.scatter(long_days, spread_win.loc[long_days], color="green", label="Long Entry", marker="^")
    ax1.scatter(short_days, spread_win.loc[short_days], color="red", label="Short Entry", marker="v")

    # Mark wipeout point
    if not wipe_dates.empty:
        wipe_day = wipe_dates[0]
        ax1.axvline(wipe_day, color="red", linestyle="--", label="Equity Wiped", linewidth=2)

    # Plot equity on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(equity_win, color="blue", alpha=0.4, label="Equity (scaled)", linestyle="--")
    ax2.set_ylabel("Equity")

    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.show()


def backtest_alpha_scaled_basket(df, beta, signal, test_start, test_end, fee_per_trade=0.001):
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
    if (equity <= 0).any():
        print("⚠️ Plotting for crash window starting at:", test_start)
        spread = df @ beta
        spread.index = pd.to_datetime(spread.index)  # Ensure index is datetime
        spread.loc[test_start:test_end].plot(title=f"Spread (Mar–Jul 2018) for test:{test_start}")
        plt.tight_layout()
        plt.show()
        ruin_index = equity[ equity <= 0 ].index[0]
        strat_ret = strat_ret.loc[:ruin_index]
        equity = equity.loc[:ruin_index]
        print(f"⚠️ Equity wiped out on {ruin_index.date()}, truncating backtest.")
        print("\n🚨 Capital fully wiped. Investigating Sharpe anomaly.")
        print(f"Equity final: {equity.iloc[-1]}")
        print(f"Sharpe: {sharpe:.2f}, Total Return: {total:.2f}, Days: {days}")
        
        print("\n📈 Last 10 equity values:")
        print(equity.tail(10))
        
        print("\n📊 strat_ret summary:")
        print(strat_ret.describe())

        print("\n🧾 strat_ret tail (last 10):")
        print(strat_ret.tail(10))

        print("\n⚠️ Large daily drops:")
        big_drops = strat_ret[strat_ret < -0.3]
        print(big_drops.tail(10))

        print("\n🔁 Position signal tail:")
        print(pos.tail(10))
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
    # ❗ Truncate on capital ruin
    if (equity <= 0).any():
        ruin_index = equity[ equity <= 0 ].index[0]
        strat_ret = strat_ret.loc[:ruin_index]
        equity = equity.loc[:ruin_index]
        print(f"⚠️ Equity wiped out on {ruin_index.date()}, truncating backtest.")
        print("\n🚨 Capital fully wiped. Investigating Sharpe anomaly.")
        print(f"Equity final: {equity.iloc[-1]}")
        print(f"Sharpe: {sharpe:.2f}, Total Return: {total:.2f}, Days: {days}")
        
        print("\n📈 Last 10 equity values:")
        print(equity.tail(10))
        
        print("\n📊 strat_ret summary:")
        print(strat_ret.describe())

        print("\n🧾 strat_ret tail (last 10):")
        print(strat_ret.tail(10))

        print("\n⚠️ Large daily drops:")
        big_drops = strat_ret[strat_ret < -0.3]
        print(big_drops.tail(10))

        print("\n🔁 Position signal tail:")
        print(pos.tail(10))

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

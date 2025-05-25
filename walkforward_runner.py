
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings("ignore", message=".*no associated frequency information.*")

from strategy_lib import (
    compute_rolling_johansen_trace,
    hybrid_signal,
    backtest_alpha_scaled_basket,
    label_clusters_kmeans,
)

def get_coint_params(df, det_order=1, k_ar_diff=2):
  model = VECM(df, k_ar_diff=2, coint_rank=1, deterministic='co')  # 'co' for NONE constant in cointegration eq
  vecm_res = model.fit()
  beta = vecm_res.beta
  alpha = vecm_res.alpha
  spread = df @ vecm_res.beta[:, 0]
  return alpha, beta, spread

def estimate_half_life(spread):
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    df = pd.DataFrame({'y': delta, 'x': lagged.loc[delta.index]})
    beta = np.linalg.lstsq(df['x'].values.reshape(-1, 1), df['y'].values, rcond=None)[0][0]
    halflife = -np.log(2) / beta
    return max(1, round(halflife))

def run_walkforward(tickers, start="2012-01-01", end="2025-01-01",
                    train_years=2, test_years=1):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close'].dropna()
    results = []

    # Setup rolling windows
    test_start = df.index.min() + relativedelta(years=train_years)
    while True:
        test_end = test_start + relativedelta(years=test_years) - pd.Timedelta(days=1)
        if test_end > df.index.max():
            break
        train_start = test_start - relativedelta(years=train_years)
        train_end = test_start - pd.Timedelta(days=1)

        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]

        # Step 1: Johansen on train window
        
        # johansen = coint_johansen(log_train, det_order=1, k_ar_diff=2)
        alpha_train, beta_train, _ = get_coint_params(train_df)
        alpha_train_mags = np.abs(alpha_train).flatten()
        alpha_train_norm = alpha_train_mags / alpha_train_mags.sum()
        beta_train_scaled = beta_train.flatten() * alpha_train_norm  # scale cointegration vector by responsiveness

        spread_test_adj = pd.Series(test_df @ beta_train_scaled, index=test_df.index)
        spread_train_adj = pd.Series(train_df @ beta_train_scaled, index=train_df.index)

        # Step 3: Get regime from trace on test window
        log_train = np.log(train_df)
        trace_df = compute_rolling_johansen_trace(log_train)
        # regime_series = compute_regime_series(trace_df)
        regime_series = label_clusters_kmeans(trace_df) 

        halflife = estimate_half_life(spread_train_adj)
        spread_train_std = spread_train_adj.std(ddof=1)
        spread_train_mean = spread_train_adj.mean()
        mult = 0.5 if halflife <= 30 else 1.0 if halflife <= 60 else 1.5
        UPPER = spread_train_mean + mult * spread_train_std
        LOWER = spread_train_mean - mult * spread_train_std
        close_threshold = 0.5 * mult * spread_train_std

        # Step 5: Hybrid signal + backtest
        sig, regime_label = hybrid_signal(spread_test_adj, regime_series, UPPER, LOWER, close_threshold)
        metrics = backtest_alpha_scaled_basket(test_df, beta_train_scaled, sig)

        metrics.update({
            'train_start': train_start.date(),
            'train_end': train_end.date(),
            'test_start': test_start.date(),
            'test_end': test_end.date(),
            'Regime':      regime_label,
        })
        results.append(metrics)

        test_start = test_end + pd.Timedelta(days=1)

    return pd.DataFrame(results)

if __name__ == "__main__":
    tickers = ['SPY', 'QQQ', 'XLK']
    df_result = run_walkforward(tickers)
    print(df_result)
    df_result.to_csv("walkforward_results.csv", index=False)


import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from strategy_lib.config import START_DATE, END_DATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore", message=".*no associated frequency information.*")

from strategy_lib import (
    compute_rolling_johansen_trace,
    hybrid_signal,
    backtest_alpha_scaled_basket,
    label_clusters_kmeans,
    engineer_features,
    mean_reversion_signal,
    trend_signal,
    backtest_alpha_scaled_basket_soft,
    predictive_regime_signal,
)
def check_cointegration(df, det_order=1, k_ar_diff=2, level=0.95):
    """
    Returns True if Johansen test detects at least one cointegrating relationship at the given level.
    """
    johansen = coint_johansen(np.log(df), det_order=det_order, k_ar_diff=k_ar_diff)

    # Get first trace stat (H0: r <= 0) and its critical value
    trace_stat = johansen.lr1[0]
    crit_idx = {0.90: 0, 0.95: 1, 0.99: 2}[level]
    crit_val = johansen.cvt[0, crit_idx]

    print(f"Trace stat: {trace_stat:.2f} | 95% Critical: {crit_val:.2f} | {'PASSED' if trace_stat > crit_val else 'REJECTED'}")

    return trace_stat > crit_val

def get_coint_params(df, det_order=1, k_ar_diff=2):
    model = VECM(df, k_ar_diff=2, coint_rank=1, deterministic='li')  # 'co' for NONE constant in cointegration eq
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

def convert_regime_counts(counts: dict) -> dict:
    label_map = {0: "mean-revert", 1: "trend", 2: "neutral"}
    return {label_map[k]: v for k, v in counts.items() if k in label_map}

def run_walkforward(tickers, start=START_DATE, end=END_DATE,
                    train_years=2, test_years=1, test_months=0):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close'].dropna()
    log_df = np.log(df)
    meta_df = pd.read_csv("data/regime_training_dataset.csv", parse_dates=["date"])
    meta_df = meta_df.set_index("date")
    meta_df = engineer_features(meta_df)
    results = []
    results_soft = []

    # Setup rolling windows
    test_start = log_df.index.min() + relativedelta(years=train_years)
    while True:
        test_end = test_start + relativedelta(years=test_years) + relativedelta(months=test_months)- pd.Timedelta(days=1)
        if test_end > log_df.index.max():
            break
        train_start = test_start - relativedelta(years=train_years)
        train_end = test_start - pd.Timedelta(days=1)

        train_log = log_df.loc[train_start:train_end]
        test_log = log_df.loc[test_start:test_end]

        train_df = df.loc[train_start:train_end]  # raw price df
        test_df = df.loc[test_start:test_end]    # raw price df
        meta_train = meta_df.loc[train_start:train_end].dropna()
        meta_test = meta_df.loc[test_start:test_end].dropna()
        meta_hist = meta_df.loc[:test_start].dropna() 
        regime_counts = meta_hist['regime_label'].value_counts(normalize=True).to_dict()
        feature_cols = [c for c in meta_train.columns if c not in ['regime_label', 'date']]
        # Hardcode correct mapping of int -> string
        label_map = {0: 'trend', 1: 'mean-revert', 2: 'neutral'}


        clf_pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                class_weight='balanced',
                random_state=42
            ))
        ])
        # Fit on train regime labels
        clf_pipe.fit(meta_train[feature_cols], meta_train['regime_label'])
        # Predict regimes in test window
        regime_preds = clf_pipe.predict(meta_test[feature_cols])
        regime_series_pred = pd.Series(regime_preds, index=meta_test.index).map(label_map)
        regime_proba_df = pd.DataFrame(clf_pipe.predict_proba(meta_test[feature_cols]),
                                        index=meta_test.index,
                                        columns=clf_pipe.named_steps['clf'].classes_)
        regime_proba_df.columns = [label_map[int(cls)] for cls in regime_proba_df.columns]

        # johansen = coint_johansen(log_train, det_order=1, k_ar_diff=2)
        alpha_train, beta_train, _ = get_coint_params(train_log)
        beta_train_scaled = beta_train.flatten()
        beta_train_scaled /= np.abs(beta_train_scaled).sum()          # ℓ¹-norm = 1  ← key line

        spread_test_adj = pd.Series(test_log @ beta_train_scaled, index=test_df.index)
        spread_train_adj = pd.Series(train_log @ beta_train_scaled, index=train_df.index)
        # Step 3: Get regime from trace on test window
        trace_df = compute_rolling_johansen_trace(train_log)
        # regime_series = compute_regime_series(trace_df)
        regime_series = label_clusters_kmeans(trace_df) 

        spread_train_std = spread_train_adj.std(ddof=1)
        spread_train_mean = spread_train_adj.mean()
        sensitivity = np.clip(np.linalg.norm(alpha_train), 0.5, 2.0)
        mult = sensitivity  # instead of using halflife
        UPPER = spread_train_mean + mult * spread_train_std
        LOWER = spread_train_mean - mult * spread_train_std
        close_threshold = 0.5 * mult * spread_train_std

        # Step 5: Hybrid signal + backtest

        sig_trend = trend_signal(spread_test_adj)
        # sig_revert = mean_reversion_signal(spread_test_adj, UPPER, LOWER, close_threshold)
        sig_revert = mean_reversion_signal(spread_test_adj, UPPER, LOWER, close_threshold, spread_train_mean)
        regime_counts = convert_regime_counts(regime_counts)
        soft_signal, regime_label_soft = predictive_regime_signal(sig_trend, sig_revert, regime_proba_df)

        sig, regime_label = hybrid_signal(spread_test_adj, regime_series, UPPER, LOWER, close_threshold, spread_train_mean)
        metrics = backtest_alpha_scaled_basket(test_log, beta_train_scaled, sig, test_start, test_end)
        metrics_soft = backtest_alpha_scaled_basket_soft(test_df, beta_train_scaled, soft_signal)
        metrics.update({
            'train_start': train_start.date(),
            'train_end': train_end.date(),
            'test_start': test_start.date(),
            'test_end': test_end.date(),
             'Regime':      regime_label,
        })
        results.append(metrics)
        metrics_soft.update({
            'train_start': train_start.date(),
            'train_end': train_end.date(),
            'test_start': test_start.date(),
            'test_end': test_end.date(),
            'regime_label': regime_label_soft,
        })
        results_soft.append(metrics_soft)

        test_start = test_end + pd.Timedelta(days=1)

    return pd.DataFrame(results), pd.DataFrame(results_soft)

if __name__ == "__main__":
    tickers = ['SPY', 'QQQ', 'XLK']
    df_result, df_result_soft = run_walkforward(tickers)
    print(df_result)
    print(df_result_soft)
    df_result.to_csv("walkforward_results_naive_regime.csv", index=False)
    print("net return:", ((df_result['Total Return'] + 1).cumprod()).iloc[-1] - 1)
    df_result_soft.to_csv("walkforward_results_predictive_regime.csv", index=False)
    print("net return:", ((df_result_soft['Total Return'] + 1).cumprod()).iloc[-1] - 1)


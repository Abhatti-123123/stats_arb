
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
    mean_reversion_signal_zscore
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

def convert_regime_counts(counts: dict) -> dict:
    label_map = {0: "mean-revert", 1: "trend", 2: "neutral"}
    return {label_map[k]: v for k, v in counts.items() if k in label_map}

def run_walkforward(tickers, start=START_DATE, end=END_DATE,
                    train_years=2, test_years=1, test_months=0):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close'].dropna()
    meta_df = pd.read_csv("data/regime_training_dataset.csv", parse_dates=["date"])
    meta_df = meta_df.set_index("date")
    meta_df = engineer_features(meta_df)
    results = []
    results_soft = []
    results_data_driven = []

    # Setup rolling windows
    test_start = df.index.min() + relativedelta(years=train_years)
    while True:
        test_end = test_start + relativedelta(years=test_years) + relativedelta(months=test_months)- pd.Timedelta(days=1)
        if test_end > df.index.max():
            break
        train_start = test_start - relativedelta(years=train_years)
        train_end = test_start - pd.Timedelta(days=1)

        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]
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
        alpha_train, beta_train, _ = get_coint_params(train_df)
        beta_train_scaled = beta_train.flatten()

        spread_test_adj = pd.Series(test_df @ beta_train_scaled, index=test_df.index)
        spread_train_adj = pd.Series(train_df @ beta_train_scaled, index=train_df.index)
        
        # Step 3: Get regime from trace on test window
        log_train = np.log(train_df)
        trace_df = compute_rolling_johansen_trace(log_train)
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
        sig_revert = mean_reversion_signal(spread_test_adj, UPPER, LOWER, close_threshold)
        regime_counts = convert_regime_counts(regime_counts)
        soft_signal, regime_label_soft = predictive_regime_signal(sig_trend, sig_revert, regime_proba_df)

        sig, regime_label = hybrid_signal(spread_test_adj, regime_series, UPPER, LOWER, close_threshold)
        metrics = backtest_alpha_scaled_basket(test_df, beta_train_scaled, sig)
        metrics_soft = backtest_alpha_scaled_basket_soft(test_df, beta_train_scaled, soft_signal)
##################################### 
        LOOKAHEAD = 5  # Number of days to evaluate signal performance

        signal_labels = []
        X_train_rows = []

        LOOKAHEAD = 5
        valid_indices = spread_train_adj.index[:-LOOKAHEAD]  # prevent out-of-bounds

        for t in valid_indices:
            future_t = spread_train_adj.index[spread_train_adj.index.get_loc(t) + LOOKAHEAD]

            # Skip if features not available at time t
            if t not in meta_train.index or future_t not in spread_train_adj.index:
                continue

            # Get forward return
            forward_ret = (spread_train_adj[future_t] / spread_train_adj[t]) - 1

            # Get signals
            trend_val = sig_trend.get(t, 0)
            revert_val = sig_revert.get(t, 0)

            pnl_trend = trend_val * forward_ret
            pnl_revert = revert_val * forward_ret

            # Label: 1 = trend better, 0 = revert better
            label = int(pnl_trend > pnl_revert)
            signal_labels.append(label)

            # Features at time t
            row = meta_train.loc[t, feature_cols]
            X_train_rows.append(row)

        X_train_df = pd.DataFrame(X_train_rows)
        y_train = pd.Series(signal_labels)
#       
# 
        # Evaluate which strategy worked better in train window
        sig_trend_train = trend_signal(spread_train_adj)
        sig_revert_train = mean_reversion_signal(spread_train_adj, UPPER, LOWER, close_threshold)

        perf_trend = backtest_alpha_scaled_basket(train_df, beta_train_scaled, sig_trend_train)
        perf_revert = backtest_alpha_scaled_basket(train_df, beta_train_scaled, sig_revert_train)

        if perf_trend['Sharpe'] > perf_revert['Sharpe'] + 0.1:
            dominant_regime = 'trend'
        elif perf_revert['Sharpe'] > perf_trend['Sharpe'] + 0.1:
            dominant_regime = 'mean-revert'
        else:
            dominant_regime = 'neutral'
# 
        signal_clf = Pipeline([
            ('scale', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                class_weight='balanced',
                random_state=42
            ))
        ])
# 
        signal_clf.fit(X_train_df, y_train)

        X_test_df = meta_test[feature_cols].copy()
        pred_signal_label = signal_clf.predict(X_test_df)

        # Build final signal
        signal_data_driven = pd.Series(
            np.where(pred_signal_label == 1, sig_trend.loc[X_test_df.index], sig_revert.loc[X_test_df.index]),
            index=X_test_df.index
        )
# 

# 
        metrics_data_driven = backtest_alpha_scaled_basket(test_df.loc[signal_data_driven.index], beta_train_scaled, signal_data_driven)
        metrics_data_driven.update({
            'train_start': train_start.date(),
            'train_end': train_end.date(),
            'test_start': test_start.date(),
            'test_end': test_end.date(),
            'Regime': 'data-driven-signal'
        })
        results_data_driven.append(metrics_data_driven)


# 
        


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

    return pd.DataFrame(results), pd.DataFrame(results_soft), pd.DataFrame(results_data_driven)

if __name__ == "__main__":
    tickers = ['SPY', 'QQQ', 'XLK']
    df_result, df_result_soft, df_results_driven = run_walkforward(tickers)
    print(df_result)
    print(df_result_soft)
    print(df_results_driven)
    df_result.to_csv("walkforward_results_naive_regime.csv", index=False)
    print("net return:", ((df_result['Total Return'] + 1).cumprod()).iloc[-1] - 1)
    df_result_soft.to_csv("walkforward_results_predictive_regime.csv", index=False)
    print("net return:", ((df_result_soft['Total Return'] + 1).cumprod()).iloc[-1] - 1)
    df_results_driven.to_csv("walkforward_results_predictive_regime.csv", index=False)
    print("net return:", ((df_results_driven['Total Return'] + 1).cumprod()).iloc[-1] - 1)


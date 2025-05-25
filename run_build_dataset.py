# run_build_dataset.py

import pandas as pd
from strategy_lib.config import (
    TRACE_STATS_FILE, FANG_WEIGHT_FILE, SPY_OPTION_FILES_GLOB,
    FINAL_DATASET_PATH, TECH_TICKERS, MACRO_TICKERS, ALL_DATA_PATH,
    START_DATE, END_DATE
)
from strategy_lib.data_loader import (
    load_marketcap_weights, load_combined_tech_iv, get_stock_data, compute_weighted_series, compute_weighted_iv
)
from strategy_lib.option_utils import load_options_data
from strategy_lib.regime_dataset_builder import build_regime_prediction_df

# --- Load inputs ---
trace_df = pd.read_csv(TRACE_STATS_FILE, parse_dates=['Date'])
trace_df.rename(columns={'r0': 'r=0', 'r1': 'r=1', 'r2': 'r=2'}, inplace=True)

# Download SPY, XLK, and tech prices
tech_prices = get_stock_data(TECH_TICKERS, START_DATE, END_DATE)
macro_prices = get_stock_data(MACRO_TICKERS, START_DATE, END_DATE)
spy_df = macro_prices[['SPY']].rename(columns={'SPY': 'Close'})
xlk_df = macro_prices[['XLK']].rename(columns={'XLK': 'Close'})

tech_iv_df = load_combined_tech_iv(ALL_DATA_PATH, TECH_TICKERS)
fang_wts_df = load_marketcap_weights(FANG_WEIGHT_FILE)
weighted_tech_iv_mean_df = compute_weighted_iv(tech_iv_df, fang_wts_df)
weighted_tech_prices = compute_weighted_series(tech_prices, fang_wts_df)
tech_return = weighted_tech_prices.pct_change()
tech_zscore = (tech_return - tech_return.rolling(20).mean()) / tech_return.rolling(20).std()

date_filter = set(trace_df['Date'].unique())
spy_options = load_options_data(SPY_OPTION_FILES_GLOB, date_filter)

# --- Build dataset ---
final_df = build_regime_prediction_df(
    tech_return_df=tech_return,
    trace_df=trace_df,
    spy_df=spy_df,
    xlk_df=xlk_df,
    tech_iv_df=weighted_tech_iv_mean_df,
    spy_options=spy_options
)

# --- Save ---
final_df.to_csv(FINAL_DATASET_PATH, index=False)
print(f"✅ Dataset saved to: {FINAL_DATASET_PATH}")

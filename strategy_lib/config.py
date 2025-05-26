RISK_FREE_RATE = 0.0425

ALL_DATA_PATH = "data/all_data/"
TECH_IV_CSVS = [
    "AAPL US Equity.csv", "MSFT US Equity.csv", "AMZN US Equity.csv",
    "GOOGL US Equity.csv", "TSLA US Equity.csv",
    "NVDA US Equity.csv", "META US Equity.csv"
]
SPY_OPTION_FILES_GLOB = "data/all_data/spy_eod_*.txt"
FANG_WEIGHT_FILE = "data/fang_weights.csv"
TRACE_STATS_FILE = "data/trace_stats_with_regimes.csv"


# Output
FINAL_DATASET_PATH = "data/regime_training_dataset.csv"

# Date range for backtest or model windowing
START_DATE = "2015-05-27"
END_DATE   = "2023-12-30"

# Expiry buckets (in days) for options IV extraction
EXPIRY_BUCKETS = [10, 20, 30]  # used in get_nearest_spy_options

# Ticker list for downloading EOD prices
TECH_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META']
MACRO_TICKERS = ['SPY', 'XLK']
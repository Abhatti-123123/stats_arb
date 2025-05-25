import pandas as pd
import numpy as np
import yfinance as yf
from strategy_lib import compute_rolling_johansen_trace
from strategy_lib.regime_labeler import label_trace_regimes
from strategy_lib.config import TRACE_STATS_FILE, START_DATE, END_DATE

def main():
    tickers = ['SPY', 'QQQ', 'XLK']

    print("📥 Downloading price data...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)['Close'].dropna()

    print("🧮 Computing rolling Johansen trace statistics...")
    log_data = np.log(data)
    trace_df = compute_rolling_johansen_trace(log_data)

    print("🏷️ Assigning regime labels using KMeans...")
    trace_df_labels = label_trace_regimes(trace_df)

    print(f"💾 Saving dataset to: {TRACE_STATS_FILE}")
    trace_df_labels.to_csv(TRACE_STATS_FILE, index=True)

    print("✅ Trace dataset generation complete.")

if __name__ == "__main__":
    main()
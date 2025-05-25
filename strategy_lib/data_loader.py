# strategy_lib/data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
import re

def get_stock_data(tickers, start, end):
    """
    Downloads price data from Yahoo Finance.
    Returns:
        - tech_prices_df: tech tickers only
        - macro_prices_df: XLK, SPY subset
    """
    data = yf.download(tickers, start=start, end=end, progress=False,
                       group_by='ticker', threads=True, auto_adjust=True)
    adj_close = data.xs('Close', level=1, axis=1)[tickers].dropna()

    # tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'TSLA', 'NVDA', 'META']
    # macro_tickers = ['SPY', 'XLK']

    df = adj_close[[t for t in tickers if t in adj_close.columns]]
    # macro_df = adj_close[[t for t in macro_tickers if t in adj_close.columns]]

    return df

def load_marketcap_weights(path: str) -> pd.DataFrame:
    """
    Loads weekly FANG weights, normalizes them, and forward-fills to daily.
    """
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = pd.to_datetime(df.index, errors='coerce')

    tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    df = df[[col for col in tickers if col in df.columns]]
    normalized = df.div(df.sum(axis=1), axis=0)
    normalized = normalized.asfreq('D').ffill()  # forward-fill to match daily frequency
    return normalized

def compute_weighted_iv(iv_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """
    Computes market-cap weighted Tech IV using fuzzy ticker matching.
    Assumes IV columns are like 'AAPL_IV', 'MSFT_IV', etc.,
    while weight columns are like 'AAPL', 'MSFT', etc.
    """
    weights_df = weights_df.reindex(iv_df.index).ffill()

    # Build a mapping: 'AAPL' -> 'AAPL_IV' etc.
    ticker_map = {}
    for iv_col in iv_df.columns:
        base = re.sub(r'_IV$', '', iv_col.upper())  # normalize to uppercase base
        if base in weights_df.columns:
            ticker_map[base] = iv_col

    if not ticker_map:
        raise ValueError("❌ No overlapping tickers between weights and IV data")

    # Extract aligned columns
    iv_matched = iv_df[[ticker_map[t] for t in ticker_map]]
    wts_matched = weights_df[list(ticker_map.keys())]

    # Weighted IV: sum(row-wise)
    weighted_iv = (iv_matched.values * wts_matched.values).sum(axis=1)
    return pd.Series(weighted_iv, index=iv_df.index, name='Tech_IV_weighted')

def compute_weighted_series(data_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    """
    Computes a market-cap weighted average series across tech tickers.

    Parameters:
    - data_df: DataFrame with columns = tickers (e.g. AAPL, MSFT...) and daily index
    - weights_df: DataFrame with matching tickers as columns, index aligned or forward-fillable

    Returns:
    - pd.Series with weighted average value per day
    """
    # Align and ffill weights
    weights_df = weights_df.reindex(data_df.index).ffill()

    common = [col for col in data_df.columns if col in weights_df.columns]
    if not common:
        raise ValueError("No common tickers between data and weights")

    # Multiply each column by its weight
    weighted = data_df[common] * weights_df[common]
    return weighted.sum(axis=1)

def load_combined_tech_iv(folder_path: str, tickers: list) -> pd.DataFrame:
    """
    Loads individual tech stock IV CSVs and computes average daily IV.
    Returns DataFrame with: Date index, columns=[tickers + 'IV_mean']
    """
    iv_data = {}
    for ticker in tickers:
        file_path = os.path.join(folder_path, f"{ticker} US Equity.csv")
        try:
            df = pd.read_csv(file_path, index_col=0)
            df.columns = df.columns.str.strip()
            df.index = pd.to_datetime(df.index, errors='coerce')
            iv_col = next((c for c in df.columns if 'iv' in c.lower()), None)
            if iv_col is None:
                raise ValueError(f"Could not find IV column in {file_path}")
            iv_data[ticker] = df[iv_col].rename(f"{ticker}_IV")
        except Exception as e:
            print(f"⚠️ Error loading {ticker}: {e}")

    if not iv_data:
        raise ValueError("No tech IV data loaded successfully.")

    combined = pd.concat(iv_data.values(), axis=1)
    return combined

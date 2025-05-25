import pandas as pd
import numpy as np
import glob

# --- Config ---
REQUIRED_COLS = [
    'QUOTE_DATE', 'UNDERLYING_LAST', 'EXPIRE_DATE',
    'C_LAST', 'C_IV', 'P_LAST', 'P_IV', 'STRIKE',
    'C_ASK', 'C_BID', 'P_ASK', 'P_BID', 'P_VEGA', 'C_VEGA'
]

DTYPES = {
    'UNDERLYING_LAST': 'float32', 'C_LAST': 'float32', 'C_IV': 'float32',
    'P_LAST': 'float32', 'P_IV': 'float32', 'STRIKE': 'float32',
    'C_ASK': 'float32', 'C_BID': 'float32', 'P_ASK': 'float32', 'P_BID': 'float32',
    'P_VEGA': 'float32', 'C_VEGA': 'float32'
}

# --- Loader ---
def load_options_data(data_path: str, date_filter: set = None) -> pd.DataFrame:
    """
    Loads SPY options with memory-efficient column subset.
    Optionally filter by QUOTE_DATE if passed.
    """
    all_files = glob.glob(data_path)
    dfs = []

    for file in all_files:
        try:
            df = pd.read_csv(
            file,
            sep=',',
            dtype=str,   # Read everything as string to minimize memory, parse later
            low_memory=False
            )
            df.columns = df.columns.str.strip().str.replace('[\\[\\]]', '', regex=True)
            df = df[[col for col in REQUIRED_COLS if col in df.columns]]
            df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'], errors='coerce')
            df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'], errors='coerce')
            if date_filter:
                df = df[df['QUOTE_DATE'].isin(date_filter)]
            for col in DTYPES:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(DTYPES[col])
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return pd.concat(dfs, ignore_index=True).sort_values(['QUOTE_DATE', 'EXPIRE_DATE'])

# --- Weighted IV ---
def vega_weighted_avg(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    bcf = {}
    for col in df.columns:
        valid = df[weight_col].notna() & df[col].notna()
        wsum = df.loc[valid, weight_col].sum()
        bcf[col] = np.nan if valid.sum() == 0 or wsum == 0 else (df.loc[valid, col] * df.loc[valid, weight_col]).sum() / wsum
    return pd.DataFrame([bcf])

# --- Option Feature Extractor ---
def get_nearest_spy_options(event_date, spy_price, spy_options, moneyness_band=0.025):
    try:
        event_date = pd.to_datetime(event_date)

        # --- Pre-slice once ---
        daily_opts = spy_options[spy_options['QUOTE_DATE'] == event_date]
        if daily_opts.empty:
            return {}, {}

        call_list = {}
        put_list = {}

        # --- Loop over expiry buckets ---
        expiry_buckets = [(1, 0, 10), (2, 10, 20), (3, 20, 30)]
        for label, start_day, end_day in expiry_buckets:
            subset = daily_opts[
                (daily_opts['EXPIRE_DATE'] > event_date + pd.Timedelta(days=start_day)) &
                (daily_opts['EXPIRE_DATE'] <= event_date + pd.Timedelta(days=end_day))
            ]
            if subset.empty:
                continue

            # ATM strike filter
            atm_strikes = subset[
                (subset['STRIKE'] >= spy_price * (1 - moneyness_band)) &
                (subset['STRIKE'] <= spy_price * (1 + moneyness_band))
            ]['STRIKE'].unique()

            # --- CALLS ---
            calls = subset[
                (subset['STRIKE'].isin(atm_strikes)) &
                (subset['C_ASK'].notna()) & (subset['C_BID'].notna())
            ].assign(
                C_MID=lambda df: (df['C_ASK'] + df['C_BID']) / 2
            )[['C_MID', 'C_IV', 'P_IV', 'C_VEGA']]

            calls['C_IV'] = calls['C_IV'].fillna(calls['P_IV'])
            calls['iv_rank'] = calls['C_IV'].rank(pct=True) * 100
            calls = calls[calls['iv_rank'].between(25, 75)]

            # --- PUTS ---
            puts = subset[
                (subset['STRIKE'].isin(atm_strikes)) &
                (subset['P_ASK'].notna()) & (subset['P_BID'].notna())
            ].assign(
                P_MID=lambda df: (df['P_ASK'] + df['P_BID']) / 2
            )[['P_MID', 'P_IV', 'C_IV', 'P_VEGA']]

            puts['P_IV'] = puts['P_IV'].fillna(puts['C_IV'])
            puts['iv_rank'] = puts['P_IV'].rank(pct=True) * 100
            puts = puts[puts['iv_rank'].between(25, 75)]

            if calls.empty or puts.empty:
                continue
            if (
                calls[['C_MID', 'C_IV']].isnull().all().any() or
                puts[['P_MID', 'P_IV']].isnull().all().any()
            ):
                continue

            # --- Vega-weighted average ---
            calls = calls.drop(columns='iv_rank', errors='ignore')
            puts = puts.drop(columns='iv_rank', errors='ignore')
            calls = vega_weighted_avg(calls, 'C_VEGA')
            puts = vega_weighted_avg(puts, 'P_VEGA')

            call_list[label] = calls
            put_list[label] = puts

        return call_list, put_list

    except Exception as e:
        print(f"SPY options error: {str(e)}")
        return {}, {}


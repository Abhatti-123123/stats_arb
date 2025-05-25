# strategy_lib/regime_dataset_builder.py

import pandas as pd
import numpy as np
from strategy_lib.option_utils import get_nearest_spy_options
from strategy_lib.config import EXPIRY_BUCKETS

def safe_extract_iv(calls: dict, expiry: int) -> float:
    try:
        return float(calls[expiry]['C_IV'].iloc[0])
    except Exception:
        return np.nan

def build_regime_prediction_df(
    tech_return_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    xlk_df: pd.DataFrame,
    tech_iv_df: pd.DataFrame,
    spy_options: pd.DataFrame
) -> pd.DataFrame:
    """
    Build full feature DataFrame for regime classification.
    Merges Johansen trace stats, SPY/XLK returns, IV surfaces, and tech features.
    """
    final_rows = []

    for idx, row in trace_df.iterrows():
        date = pd.to_datetime(row['Date'])
        try:
            # --- Trace stats ---
            r0, r1, r2 = row['r=0'], row['r=1'], row['r=2']
            gap_01, gap_12 = r0 - r1, r1 - r2
            regime = row['regime']

            # --- Prices ---
            if date not in spy_df.index or date not in xlk_df.index:
                continue
            spy_price = spy_df.loc[date]['Close']
            xlk_today = xlk_df.loc[date]['Close']
            xlk_yest = xlk_df.shift(1).loc[date]['Close']
            xlk_return = (xlk_today - xlk_yest) / xlk_yest if xlk_yest != 0 else np.nan

            # --- Correlation ---
            window = 20
            spy_ret = np.log(spy_df['Close']).diff()
            xlk_ret = np.log(xlk_df['Close']).diff()
            combined = pd.concat([spy_ret, xlk_ret], axis=1, join='inner')
            combined.columns = ['spy', 'xlk']
            corr = combined.loc[:date].tail(window).corr().iloc[0, 1]

            # --- SPY Options IVs ---
            calls, puts = get_nearest_spy_options(date, spy_price, spy_options)
            spy_call_iv_10 = safe_extract_iv(calls, 1)
            spy_call_iv_20 = safe_extract_iv(calls, 2)
            spy_call_iv_30 = safe_extract_iv(calls, 3)
            spy_put_iv_10 = safe_extract_iv(puts, 1)
            spy_put_iv_20 = safe_extract_iv(puts, 2)
            spy_put_iv_30 = safe_extract_iv(puts, 3)

            # --- Append row ---
            final_rows.append({
                'date': date,
                'trace_r0': r0,
                'trace_r1': r1,
                'trace_r2': r2,
                'trace_gap_01': gap_01,
                'trace_gap_12': gap_12,
                'XLK_return': xlk_return,
                'XLK_SPY_corr': corr,
                'SPY_CALL_IV_10': spy_call_iv_10,
                'SPY_CALL_IV_20': spy_call_iv_20,
                'SPY_CALL_IV_30': spy_call_iv_30,
                'SPY_PUT_IV_10': spy_put_iv_10,
                'SPY_PUT_IV_20': spy_put_iv_20,
                'SPY_PUT_IV_30': spy_put_iv_30,
                'Tech_IV': tech_iv_df.get(date, np.nan),
                'Tech df': tech_return_df.get(date, np.nan),
                'regime': regime
            })

        except Exception as e:
            print(f"⚠️ Row error at {date}: {e}")
            continue

    return pd.DataFrame(final_rows)

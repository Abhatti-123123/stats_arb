import pandas as pd

def engineer_features(df: pd.DataFrame, drop_redundant=True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 1. Fill missing CALL/PUT IVs with counterpart value
    for t in [10, 20, 30]:
        call_col = f"SPY_CALL_IV_{t}"
        put_col = f"SPY_PUT_IV_{t}"

        df[call_col] = df[call_col].fillna(df[put_col])
        df[put_col] = df[put_col].fillna(df[call_col])

    # 2. IV Term Structure Slope (expect: positive → mean reversion, negative → trend panic)
    df["IV_slope_10_30"] = df["SPY_CALL_IV_30"] - df["SPY_CALL_IV_10"]

    # 3. IV Skew (risk reversal): PUT_IV - CALL_IV
    for t in [10, 20, 30]:
        df[f"PUT_CALL_skew_{t}"] = df[f"SPY_PUT_IV_{t}"] - df[f"SPY_CALL_IV_{t}"]

    # 4. Tech IV Diff (momentum proxy) 
    # df["Tech_IV_diff"] = df["Tech_IV"].diff().fillna(0)

    # 5. XLK return magnitude (vol regime signal)
    df["XLK_return_abs"] = df["XLK_return"].abs()

    if drop_redundant:
        drop_cols = [
            "SPY_CALL_IV_10", "SPY_CALL_IV_30",
            "SPY_PUT_IV_10", "SPY_PUT_IV_30",
            "Tech_IV", "XLK_return"
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 6. Ffill remaining missing values
    df = df.ffill()
    print(df.isna().sum().sort_values(ascending=False))

    return df

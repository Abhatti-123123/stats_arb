import pandas as pd
import numpy as np
from granger_analysis import run_granger_causality_tests
from feature_engineering import engineer_features

df = pd.read_csv("../data/regime_training_dataset.csv")  # or your actual path
df = engineer_features(df)

granger_df = run_granger_causality_tests(df, max_lag=3, regime_class="trend")
print(granger_df)


# Lag                          1         2         3
# Feature
# IV_slope_10_30    5.239145e-07  0.000795  0.003057
# trace_gap_12      7.207967e-04  0.002427  0.005334
# trace_gap_01      2.358117e-03  0.007555  0.015883
# trace_r1          8.884855e-03  0.027477  0.056057
# XLK_return_abs    1.615589e-02  0.158046  0.480700
# trace_r0          1.926189e-02  0.056937  0.111334
# SPY_PUT_IV_20     3.318258e-02  0.070503  0.109315
# SPY_CALL_IV_20    3.394024e-02  0.097966  0.139419
# PUT_CALL_skew_10  4.919939e-01  0.504114  0.374543
# PUT_CALL_skew_30  5.384745e-01  0.035631  0.037237
# XLK_SPY_corr      6.215707e-01  0.526252  0.679628
# Tech_IV_diff      7.915656e-01  0.919047  0.939504
# Tech df           8.867255e-01  0.885710  0.814251
# trace_r2          9.575323e-01  0.996683  0.999702
# PUT_CALL_skew_20  9.668904e-01  0.727144  0.716415

# ✅ Top Predictive Features (p < 0.01 across early lags)
# Feature	Lag 1	Lag 2	Lag 3	Comments
# IV_slope_10_30	5e-07	0.0008	0.0031	📈 Very strong — term structure of IV is a regime predictor
# trace_gap_12	0.0007	0.0024	0.0053	📉 Highly predictive — gap between 2nd and 1st eigenvalues
# trace_gap_01	0.0023	0.0076	0.0159	📉 Strong — signal of cointegration decay
# trace_r1	0.0089	0.0275	0.0561	📉 Moderate power — part of trace stats
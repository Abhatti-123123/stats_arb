import pandas as pd
import numpy as np
from granger_analysis import run_granger_causality_tests
from feature_engineering import engineer_features

df = pd.read_csv("../data/regime_training_dataset.csv")  # or your actual path
df = engineer_features(df)

granger_df = run_granger_causality_tests(df, max_lag=3, regime_class="trend")
print(granger_df)


#Output:
# Lag                      1         2         3
# Feature
# Tech df           0.018078  0.016193  0.046069
# trace_r2          0.126467  0.283528  0.435865
# trace_gap_12      0.247719  0.489030  0.674210
# XLK_SPY_corr      0.268607  0.526999  0.601910
# trace_r1          0.302017  0.561011  0.739221
# PUT_CALL_skew_10  0.530386  0.357459  0.573557
# trace_gap_01      0.535379  0.818383  0.936318
# PUT_CALL_skew_30  0.536086  0.001518  0.003266
# trace_r0          0.544945  0.822304  0.936386
# SPY_PUT_IV_20     0.555894  0.699700  0.385829
# SPY_CALL_IV_20    0.559419  0.374821  0.295957
# XLK_return_abs    0.629434  0.695152  0.822556
# Tech_IV_diff      0.719991  0.006533  0.010268
# IV_slope_10_30    0.838426  0.813429  0.950273
# PUT_CALL_skew_20  0.969293  0.001120  0.000886

# Feature	Strong Causality?	Interpretation
# Tech df	✅ Yes (p < 0.05)	Lagged change in Tech IV predicts regime shift to trend
# PUT_CALL_skew_30	✅ Yes (Lag 2–3)	Risk reversal (longer-dated) has strong forward predictive power
# Tech_IV_diff	✅ Yes (Lag 2–3)	Momentum in Tech IV contributes to future regime
# PUT_CALL_skew_20	✅ Yes (Lag 2–3)	Shorter skew signal, also highly predictive
# selected_features = ["Tech df", "PUT_CALL_skew_30", "Tech_IV_diff", "PUT_CALL_skew_20"]
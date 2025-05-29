## 📊 Strategy Comparison: Mean-Reverting vs. Trend-Following vs. Regime-Switching

This analysis evaluates three trading strategies using a rolling Johansen cointegration framework. Each strategy was backtested with **0.1% slippage** applied on all trade entries, ensuring realistic execution costs.

### ✅ Strategies

- **Mean-Reverting**: Trades on cointegrated spread reversion
- **Trend-Following**: Trades in direction of spread drift when cointegration weakens
- **Regime-Switching (Walk-Forward)**: Dynamically switches between the above based on trace statistic regimes:
  - `mean-revert` if strong cointegration
  - `trend` if weak cointegration
  - `neutral` otherwise (flat or minimal allocation)

---

### 📈 Performance Summary

| Metric                | Mean-Reverting | Trend-Following | Regime-Switching |
|-----------------------|----------------|------------------|------------------|
| **Avg Sharpe**        | -3.46          | -0.81            | **-0.15** ✅     |
| **Max Sharpe**        | 1.32           | **1.85**         | **1.85** ✅     |
| **Avg Total Return**  | 0.017          | 0.41             | **0.48** ✅     |
| **Max Total Return**  | 1.26           | **4.59**         | **4.59** ✅     |

> ⚠️ All results include **0.1% slippage per entry** to reflect realistic trading conditions.

---

### 🔍 Key Insights

- **Mean-Reverting strategy** had one exceptional year (2020), but failed in most periods.
- **Trend-Following strategy** offered slightly more consistent behavior, but still underperformed on average.
- **Regime-Switching strategy** consistently outperformed both, despite slippage — validating the hypothesis that:
  - Cointegration strength (Trace Statistic) predicts which strategy is appropriate.
  - Regime adaptation with clustering (kmeans) smooths volatility and enhances risk-adjusted returns.

---

### 💡 Future Work

- **Enhance "neutral" regime modeling**: Currently underperforming, possibly masking additional structure.
- **Introduce macro filters or volatility-based features** to improve switching accuracy.
- **Classify market states beyond trace stats** using clustering or causal inference (e.g., PC algorithm, Granger).

Latest results:
1) using regime of train walk
2) using regime switch based on macros, option ivs etc (Great results, working on confirming the alpha on bigger dataset and regressing on factors to confirm)
3) Running a decision tree on a to maximize profits
![image](https://github.com/user-attachments/assets/30d4b022-5cde-45d6-9e47-d074202c0fae)


Can't train on past data because tech started having large weightage in spy after 2015.
This also shows we are arbitraging tech heavy market inefficency in spy.





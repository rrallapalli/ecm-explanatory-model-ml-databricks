# Databricks notebook source
# DBTITLE 1,Model Diagnostics Header
# MAGIC %md
# MAGIC # 08 Model Diagnostics
# MAGIC
# MAGIC ## Overview
# MAGIC Comprehensive model diagnostics across 8 dimensions to assess prediction quality, stability, bias, and regime-specific performance. Identifies where the model excels, where it struggles, and whether prediction errors are systematic or random.
# MAGIC
# MAGIC ## Purpose
# MAGIC
# MAGIC **Why diagnose?**
# MAGIC * **Bias detection**: Does the model systematically over/under-predict?
# MAGIC * **Regime performance**: Does accuracy degrade in specific market conditions (negative, flat, positive)?
# MAGIC * **Tail risk**: Can the model predict extreme return periods (critical for risk management)?
# MAGIC * **Temporal stability**: Is directional accuracy stable over time or degrading?
# MAGIC * **Error structure**: Are residuals random (good) or autocorrelated (missing signal)?
# MAGIC
# MAGIC **When to run:**
# MAGIC * After every model training cycle
# MAGIC * Before model deployment
# MAGIC * Monthly in production for performance monitoring
# MAGIC * When investigating unexpected trading signals or losses
# MAGIC
# MAGIC ## Diagnostic Categories
# MAGIC
# MAGIC ### 1. Core Metrics
# MAGIC
# MAGIC **Standard evaluation metrics**:
# MAGIC * **MAE** (Mean Absolute Error) - average prediction error magnitude
# MAGIC * **RMSE** (Root Mean Squared Error) - penalizes large errors
# MAGIC * **R²** - explained variance (0-1, higher is better)
# MAGIC * **Directional Accuracy** - % correct sign predictions (most critical for trading)
# MAGIC * **Information Coefficient (IC)** - Spearman rank correlation (actuals vs predictions)
# MAGIC * **IC p-value** - statistical significance of IC
# MAGIC
# MAGIC **Saved to**: `gold_model_metrics`
# MAGIC
# MAGIC ### 2. Bias Testing
# MAGIC
# MAGIC **Method**: One-sample t-test on residuals (H₀: mean residual = 0)
# MAGIC
# MAGIC **Metrics computed**:
# MAGIC * **Mean residual** - average prediction error (should be ≈ 0)
# MAGIC * **Residual std** - prediction error variability
# MAGIC * **Bias t-stat** - test statistic
# MAGIC * **Bias p-value** - significance (p > 0.05 = unbiased)
# MAGIC * **Residual_unbiased** - boolean flag
# MAGIC
# MAGIC **Why it matters**:
# MAGIC * **Positive mean residual** = systematic under-prediction (model too conservative)
# MAGIC * **Negative mean residual** = systematic over-prediction (model too optimistic)
# MAGIC * **p-value < 0.05** = bias is statistically significant, not just noise
# MAGIC
# MAGIC **Directionality breakdown**:
# MAGIC * **Under-prediction** - residual > 0 (actual higher than predicted)
# MAGIC * **Over-prediction** - residual < 0 (actual lower than predicted)
# MAGIC * % of each saved to `gold_residual_bias_summary`
# MAGIC
# MAGIC ### 3. Return Regime Analysis
# MAGIC
# MAGIC **Regime classification** (based on actual 3M forward log returns):
# MAGIC * **Negative**: log return < -5% (≈ -4.9% simple return)
# MAGIC * **Flat**: log return between -5% and +5%
# MAGIC * **Positive**: log return > +5%
# MAGIC
# MAGIC **Per-regime diagnostics**:
# MAGIC * Count (sample size per regime)
# MAGIC * Avg actual return
# MAGIC * Avg predicted return
# MAGIC * Avg absolute error (MAE)
# MAGIC * Directional accuracy
# MAGIC * % under-prediction
# MAGIC * Avg prediction bias (predicted - actual)
# MAGIC
# MAGIC **Key questions**:
# MAGIC * Does model maintain >50% directional accuracy across all regimes?
# MAGIC * Does model systematically under-predict rallies or over-predict crashes?
# MAGIC * Is error magnitude stable or does it spike in certain regimes?
# MAGIC
# MAGIC **Saved to**: `gold_regime_diagnostics`
# MAGIC
# MAGIC ### 4. Magnitude Regime Analysis
# MAGIC
# MAGIC **Regime classification** (based on absolute magnitude of actual move):
# MAGIC * **Small**: |log return| < 3%
# MAGIC * **Medium**: |log return| 3-8%
# MAGIC * **Large**: |log return| > 8%
# MAGIC
# MAGIC **Per-magnitude diagnostics**:
# MAGIC * Count
# MAGIC * Avg absolute actual return (magnitude)
# MAGIC * Avg absolute error (MAE)
# MAGIC * Directional accuracy
# MAGIC * RMSE
# MAGIC * **Error-to-move ratio**: MAE / avg_abs_actual
# MAGIC
# MAGIC **Interpretation**:
# MAGIC * **Error-to-move ratio < 1.0** = error is smaller than typical move (good)
# MAGIC * **Ratio increases with magnitude** = model struggles on large moves (common)
# MAGIC * **Directional accuracy degrades with magnitude** = model loses signal in volatile periods
# MAGIC
# MAGIC **Saved to**: `gold_magnitude_diagnostics`
# MAGIC
# MAGIC ### 5. Rolling Directional Accuracy
# MAGIC
# MAGIC **Purpose**: Assess temporal stability of model performance
# MAGIC
# MAGIC **Windows computed**:
# MAGIC * **6-month rolling** - captures short-term performance shifts
# MAGIC * **12-month rolling** - captures longer-term trends
# MAGIC
# MAGIC **Metrics tracked**:
# MAGIC * Mean rolling accuracy
# MAGIC * Min/max rolling accuracy
# MAGIC * Periods where rolling accuracy < 0.5 (worse than random)
# MAGIC
# MAGIC **Red flags**:
# MAGIC * **Declining trend** = model degrading over time (feature drift, regime shift)
# MAGIC * **High volatility** = unstable predictions, sensitive to noise
# MAGIC * **Extended periods < 0.5** = model has lost signal
# MAGIC
# MAGIC **Saved to**: `gold_rolling_accuracy`
# MAGIC
# MAGIC ### 6. Tail Error Analysis
# MAGIC
# MAGIC **Definition**: Performance on extreme return periods (bottom 10% and top 10%)
# MAGIC
# MAGIC **Why tails matter**:
# MAGIC * Most important for risk management and capital preservation
# MAGIC * Profitable strategies often require getting tails right
# MAGIC * Tail periods = regime changes, crisis events, policy shocks
# MAGIC
# MAGIC **Per-tail diagnostics**:
# MAGIC * Count (should be ~2 observations each for 20-row test set)
# MAGIC * Avg actual return
# MAGIC * Avg predicted return
# MAGIC * Avg absolute error
# MAGIC * Directional accuracy
# MAGIC * Avg prediction bias
# MAGIC
# MAGIC **Ideal tail performance**:
# MAGIC * **High directional accuracy** (>0.6) = model captures turning points
# MAGIC * **Low bias** = model doesn't systematically miss tail magnitude
# MAGIC * **Similar performance both tails** = model handles crashes and rallies equally well
# MAGIC
# MAGIC **Poor tail performance**:
# MAGIC * **Low directional accuracy** (<0.5) = model fails at extremes (risk management issue)
# MAGIC * **Large positive bias in bottom tail** = under-predicts crashes (dangerous)
# MAGIC * **Large negative bias in top tail** = under-predicts rallies (opportunity cost)
# MAGIC
# MAGIC **Saved to**: `gold_tail_diagnostics`
# MAGIC
# MAGIC ### 7. Residual Autocorrelation
# MAGIC
# MAGIC **Purpose**: Detect serial structure in prediction errors
# MAGIC
# MAGIC **Method**: Compute autocorrelation at lags 1, 3, 6 months
# MAGIC
# MAGIC **Interpretation**:
# MAGIC * **|AC| < 0.2** = residuals are random (good) ✓
# MAGIC * **|AC| > 0.2** = structure remaining (model missing serial signal) ⚠
# MAGIC * **Positive AC** = errors persist (over-prediction followed by over-prediction)
# MAGIC * **Negative AC** = errors mean-revert (oscillating errors)
# MAGIC
# MAGIC **Why it matters**:
# MAGIC * **Clean residuals** = model has extracted all available signal
# MAGIC * **Autocorrelated residuals** = model is missing temporal dynamics
# MAGIC * Suggests need for lagged target, momentum features, or AR terms
# MAGIC
# MAGIC **Output**: Console diagnostic (not saved to table)
# MAGIC
# MAGIC ### 8. Enhanced Diagnostics Table
# MAGIC
# MAGIC **Augmented predictions** with derived columns:
# MAGIC * `Absolute_Error` - |residual|
# MAGIC * `Squared_Error` - residual²
# MAGIC * `Actual_Direction` - 1 if actual > 0, else 0
# MAGIC * `Predicted_Direction` - 1 if predicted > 0, else 0
# MAGIC * `Direction_Correct` - 1 if directions match, else 0
# MAGIC * `Residual_Bias` - "Under_Prediction" or "Over_Prediction"
# MAGIC * `Actual_Return_Regime` - Negative, Flat, Positive
# MAGIC * `Magnitude_Regime` - Small, Medium, Large
# MAGIC
# MAGIC **Saved to**: `gold_model_diagnostics`
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC Seven gold-layer diagnostic tables:
# MAGIC
# MAGIC | Table | Contents | Use Case |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_model_diagnostics` | Row-level predictions + diagnostic columns | Drill-down analysis, time-series plots |
# MAGIC | `gold_model_metrics` | Core metrics + bias test results | Model comparison, performance tracking |
# MAGIC | `gold_residual_bias_summary` | Under vs over-prediction breakdown | Bias diagnosis |
# MAGIC | `gold_regime_diagnostics` | Performance by return regime (Neg/Flat/Pos) | Regime-specific evaluation |
# MAGIC | `gold_magnitude_diagnostics` | Performance by move magnitude (Small/Med/Large) | Volatility impact assessment |
# MAGIC | `gold_rolling_accuracy` | 6M and 12M rolling directional accuracy | Temporal stability monitoring |
# MAGIC | `gold_tail_diagnostics` | Performance on extreme return periods (top/bottom 10%) | Risk management evaluation |
# MAGIC
# MAGIC ## Interpretation Guide
# MAGIC
# MAGIC ### Healthy Model Profile
# MAGIC
# MAGIC ✓ **Core metrics**:
# MAGIC * Directional accuracy > 0.55 (consistently above random)
# MAGIC * IC > 0.3 with p < 0.05 (statistically significant)
# MAGIC * RMSE < 0.10 for 3-month log returns
# MAGIC
# MAGIC ✓ **Bias**:
# MAGIC * Mean residual ≈ 0 (within ±0.01)
# MAGIC * Bias t-test p-value > 0.05 (unbiased)
# MAGIC * Under/over-prediction split 45-55%
# MAGIC
# MAGIC ✓ **Regimes**:
# MAGIC * Directional accuracy > 0.5 in all three regimes
# MAGIC * Error magnitude stable across regimes (MAE variance < 2×)
# MAGIC * No systematic bias in any regime
# MAGIC
# MAGIC ✓ **Tails**:
# MAGIC * Directional accuracy > 0.5 in both tails
# MAGIC * Tail bias < 0.05 (doesn't systematically miss extremes)
# MAGIC
# MAGIC ✓ **Temporal stability**:
# MAGIC * Rolling accuracy mean > 0.55
# MAGIC * Rolling accuracy min > 0.4 (no extended weak periods)
# MAGIC * Stable trend (no degradation)
# MAGIC
# MAGIC ✓ **Residuals**:
# MAGIC * All autocorrelations |AC| < 0.2 (random)
# MAGIC
# MAGIC ### Warning Signs
# MAGIC
# MAGIC ⚠ **Systematic bias**:
# MAGIC * Bias t-test p < 0.05
# MAGIC * Mean residual > ±0.02
# MAGIC * Action: Check feature scaling, target leakage, sample period selection
# MAGIC
# MAGIC ⚠ **Regime failure**:
# MAGIC * Directional accuracy < 0.5 in any regime
# MAGIC * Large bias in negative regime (crash risk)
# MAGIC * Action: Add regime-specific features, consider regime-switching model
# MAGIC
# MAGIC ⚠ **Tail failure**:
# MAGIC * Directional accuracy < 0.5 in bottom tail (can't predict crashes)
# MAGIC * Action: Add tail-risk features (VIX, credit spreads), consider quantile regression
# MAGIC
# MAGIC ⚠ **Degradation**:
# MAGIC * Rolling accuracy declining trend
# MAGIC * Extended periods < 0.5
# MAGIC * Action: Feature drift investigation, model retraining, regime shift analysis
# MAGIC
# MAGIC ⚠ **Structure remaining**:
# MAGIC * |AC| > 0.3 at any lag
# MAGIC * Action: Add lagged target, momentum features, or AR model
# MAGIC
# MAGIC ## Key Metrics to Watch
# MAGIC
# MAGIC **Production monitoring dashboard**:
# MAGIC 1. **Directional accuracy** (current, 6M rolling, 12M rolling)
# MAGIC 2. **IC** (current + p-value)
# MAGIC 3. **Bias t-test p-value** (should stay > 0.05)
# MAGIC 4. **Regime breakdown** (directional accuracy by Neg/Flat/Pos)
# MAGIC 5. **Tail performance** (directional accuracy bottom 10% and top 10%)
# MAGIC 6. **Residual AC lag-1** (should stay < 0.2)
# MAGIC
# MAGIC **Alert thresholds**:
# MAGIC * Directional accuracy < 0.5 for 3+ consecutive months
# MAGIC * IC p-value > 0.10 (losing significance)
# MAGIC * Bias p-value < 0.05 (bias detected)
# MAGIC * Any regime directional accuracy < 0.4
# MAGIC * Bottom tail directional accuracy < 0.3 (crash prediction failure)

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,1. Data Loading & Core Diagnostics
# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*60)
print("1. Data Loading & Core Diagnostics Column Creation")
print("="*60)

# Load Predictions
predictions_df = (
    spark.table(f"{catalog}.{schema}.gold_model_predictions")
    .toPandas()
    .sort_values("Date")
    .reset_index(drop=True)
)

model_name = predictions_df["model_name"].iloc[0]
print(f"\nModel:           {model_name}")
print(f"Prediction rows: {len(predictions_df)}")
print(f"Date range:      {predictions_df['Date'].iloc[0]} → {predictions_df['Date'].iloc[-1]}")

# Validate column names match new pipeline
expected_cols = [
    "Date",
    "Actual_3M_Forward_Log_Return",
    "Predicted_3M_Forward_Log_Return",
    "Residual",
    "model_name",
]
missing = [c for c in expected_cols if c not in predictions_df.columns]
if missing:
    raise ValueError(f"Missing columns in predictions table: {missing}")
print("Schema validated.")

# Core Diagnostic Columns
print("\nAdding diagnostic columns...")
predictions_df["Absolute_Error"]  = predictions_df["Residual"].abs()
predictions_df["Squared_Error"]   = predictions_df["Residual"] ** 2

# Direction
predictions_df["Actual_Direction"]    = np.where(
    predictions_df["Actual_3M_Forward_Log_Return"] > 0, 1, 0
)
predictions_df["Predicted_Direction"] = np.where(
    predictions_df["Predicted_3M_Forward_Log_Return"] > 0, 1, 0
)
predictions_df["Direction_Correct"]   = np.where(
    predictions_df["Actual_Direction"] == predictions_df["Predicted_Direction"], 1, 0
)

# Bias direction
predictions_df["Residual_Bias"] = np.where(
    predictions_df["Residual"] > 0,
    "Under_Prediction",   # model predicted lower than actual
    "Over_Prediction"     # model predicted higher than actual
)

# Return regime — using log return thresholds
# ±5% log return ≈ ±4.9% simple return — appropriate for 3M horizon
predictions_df["Actual_Return_Regime"] = pd.cut(
    predictions_df["Actual_3M_Forward_Log_Return"],
    bins=[-np.inf, -0.05, 0.05, np.inf],
    labels=["Negative", "Flat", "Positive"]
).astype(str)

# Magnitude regime — how large was the actual move
predictions_df["Magnitude_Regime"] = pd.cut(
    predictions_df["Actual_3M_Forward_Log_Return"].abs(),
    bins=[0, 0.03, 0.08, np.inf],
    labels=["Small (<3%)", "Medium (3-8%)", "Large (>8%)"]
).astype(str)

print("\nEnhanced predictions table with diagnostic columns:")
display(predictions_df)

spark.createDataFrame(predictions_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_diagnostics"
)
print(f"\n✓ Diagnostics saved: {catalog}.{schema}.gold_model_diagnostics")

# COMMAND ----------

# DBTITLE 1,2. Core Metrics & Bias Testing
print("="*60)
print("2. Core Metrics & Bias Testing")
print("="*60)

# Core Metrics
actual    = predictions_df["Actual_3M_Forward_Log_Return"]
predicted = predictions_df["Predicted_3M_Forward_Log_Return"]

mae      = mean_absolute_error(actual, predicted)
rmse     = np.sqrt(mean_squared_error(actual, predicted))
r2       = r2_score(actual, predicted)
dir_acc  = predictions_df["Direction_Correct"].mean()
ic, pval = stats.spearmanr(actual, predicted)

# Bias test — t-test on residuals (should be near zero)
t_stat, t_pval = stats.ttest_1samp(predictions_df["Residual"], 0)

metrics_df = pd.DataFrame([{
    "model_name":              model_name,
    "target":                  target_col,
    "n_predictions":           len(predictions_df),
    "mae":                     round(mae, 6),
    "rmse":                    round(rmse, 6),
    "r2":                      round(r2, 4),
    "directional_accuracy":    round(dir_acc, 4),
    "information_coefficient": round(ic, 4),
    "ic_pvalue":               round(pval, 4),
    "mean_residual":           round(predictions_df["Residual"].mean(), 6),
    "residual_std":            round(predictions_df["Residual"].std(), 6),
    "residual_bias_tstat":     round(t_stat, 4),
    "residual_bias_pvalue":    round(t_pval, 4),
    "residual_unbiased":       t_pval > 0.05,  # True = no significant bias
    "pct_underprediction":     round((predictions_df["Residual_Bias"] == "Under_Prediction").mean(), 4),
}])

print("\nCore metrics:")
display(metrics_df)

print(f"\nBias interpretation:")
print(f"  Mean residual:  {predictions_df['Residual'].mean():.6f}")
if predictions_df['Residual'].mean() > 0.01:
    print(f"    → Model systematically UNDER-predicts (too conservative)")
elif predictions_df['Residual'].mean() < -0.01:
    print(f"    → Model systematically OVER-predicts (too optimistic)")
else:
    print(f"    → No systematic bias")

print(f"  Bias p-value:   {t_pval:.4f}")
if t_pval > 0.05:
    print(f"    ✓ Model is statistically unbiased")
else:
    print(f"    ⚠ Model has statistically significant bias")

spark.createDataFrame(metrics_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_metrics"
)
print(f"\n✓ Metrics saved: {catalog}.{schema}.gold_model_metrics")

# Residual Bias Summary
print(f"\n{'='*60}")
print(f"Residual Bias Breakdown")
print(f"{'='*60}")

bias_summary = (
    predictions_df
    .groupby("Residual_Bias")
    .agg(
        count=("Residual", "count"),
        avg_residual=("Residual", "mean"),
        avg_abs_error=("Absolute_Error", "mean"),
        max_abs_error=("Absolute_Error", "max"),
    )
    .reset_index()
)
bias_summary["pct_of_predictions"] = (
    bias_summary["count"] / bias_summary["count"].sum() * 100
).round(2)

print("\nUnder vs over-prediction:")
display(bias_summary)

print("\nInterpretation:")
under_pct = bias_summary[bias_summary["Residual_Bias"] == "Under_Prediction"]["pct_of_predictions"].values
if len(under_pct) > 0:
    under_pct = under_pct[0]
    if 45 <= under_pct <= 55:
        print(f"  ✓ Balanced split ({under_pct:.1f}% under, {100-under_pct:.1f}% over)")
    elif under_pct > 60:
        print(f"  ⚠ Model under-predicts {under_pct:.1f}% of the time (too conservative)")
    else:
        print(f"  ⚠ Model over-predicts {100-under_pct:.1f}% of the time (too optimistic)")

spark.createDataFrame(bias_summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_residual_bias_summary"
)
print(f"\n✓ Bias summary saved: {catalog}.{schema}.gold_residual_bias_summary")

# COMMAND ----------

# DBTITLE 1,3. Regime Analysis
print("="*60)
print("3. Return Regime Analysis")
print("="*60)

# Regime Error Analysis
regime_error = (
    predictions_df
    .groupby("Actual_Return_Regime")
    .agg(
        count=("Actual_3M_Forward_Log_Return", "count"),
        avg_actual=("Actual_3M_Forward_Log_Return", "mean"),
        avg_predicted=("Predicted_3M_Forward_Log_Return", "mean"),
        avg_abs_error=("Absolute_Error", "mean"),
        directional_accuracy=("Direction_Correct", "mean"),
        pct_underprediction=(
            "Residual_Bias",
            lambda x: (x == "Under_Prediction").mean()
        ),
    )
    .reset_index()
    .sort_values("Actual_Return_Regime")
)
regime_error["avg_prediction_bias"] = (
    regime_error["avg_predicted"] - regime_error["avg_actual"]
)

print("\nPerformance by return regime (Negative < -5%, Flat -5% to +5%, Positive > +5%):")
display(regime_error)

print("\nRegime diagnostics:")
for _, row in regime_error.iterrows():
    regime = row["Actual_Return_Regime"]
    dir_acc = row["directional_accuracy"]
    bias = row["avg_prediction_bias"]
    
    status = "✓" if dir_acc > 0.5 else "⚠"
    print(f"{status} {regime:<10} — DirAcc: {dir_acc:.4f}, Bias: {bias:+.6f}, n={int(row['count'])}")
    
    if dir_acc < 0.5:
        print(f"    WARNING: Model loses signal in {regime} regime")
    if abs(bias) > 0.05:
        if bias > 0:
            print(f"    WARNING: Model over-predicts {regime} returns")
        else:
            print(f"    WARNING: Model under-predicts {regime} returns")

spark.createDataFrame(regime_error).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_regime_diagnostics"
)
print(f"\n✓ Regime diagnostics saved: {catalog}.{schema}.gold_regime_diagnostics")

# Magnitude Regime Analysis
print(f"\n{'='*60}")
print(f"Magnitude Regime Analysis")
print(f"{'='*60}")

magnitude_error = (
    predictions_df
    .groupby("Magnitude_Regime")
    .agg(
        count=("Actual_3M_Forward_Log_Return", "count"),
        avg_abs_actual=("Actual_3M_Forward_Log_Return", lambda x: x.abs().mean()),
        avg_abs_error=("Absolute_Error", "mean"),
        directional_accuracy=("Direction_Correct", "mean"),
        rmse=("Squared_Error", lambda x: np.sqrt(x.mean())),
    )
    .reset_index()
)
magnitude_error["error_to_move_ratio"] = (
    magnitude_error["avg_abs_error"] / magnitude_error["avg_abs_actual"]
).round(4)

print("\nPerformance by magnitude of actual move:")
print("  error_to_move_ratio < 1.0 means error is smaller than the actual move\n")
display(magnitude_error)

print("\nMagnitude diagnostics:")
for _, row in magnitude_error.iterrows():
    mag = row["Magnitude_Regime"]
    ratio = row["error_to_move_ratio"]
    dir_acc = row["directional_accuracy"]
    
    status = "✓" if ratio < 1.0 else "⚠"
    print(f"{status} {mag:<18} — Error/Move: {ratio:.2f}, DirAcc: {dir_acc:.4f}, n={int(row['count'])}")
    
    if ratio > 1.5:
        print(f"    WARNING: Error magnitude exceeds actual move size")

spark.createDataFrame(magnitude_error).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_magnitude_diagnostics"
)
print(f"\n✓ Magnitude diagnostics saved: {catalog}.{schema}.gold_magnitude_diagnostics")

# COMMAND ----------

# DBTITLE 1,4. Rolling Accuracy Analysis
print("="*60)
print("4. Rolling Directional Accuracy")
print("="*60)

# Rolling Directional Accuracy
# 6-month rolling window — is directional accuracy stable over time?
predictions_df["Rolling_Dir_Acc_6M"] = (
    predictions_df["Direction_Correct"]
    .rolling(window=6, min_periods=3)
    .mean()
)
predictions_df["Rolling_Dir_Acc_12M"] = (
    predictions_df["Direction_Correct"]
    .rolling(window=12, min_periods=6)
    .mean()
)

rolling_df = predictions_df[[
    "Date",
    "Direction_Correct",
    "Rolling_Dir_Acc_6M",
    "Rolling_Dir_Acc_12M",
    "Actual_Return_Regime",
]].copy()

print(f"\nRolling directional accuracy (6M window):")
print(f"  Mean:               {rolling_df['Rolling_Dir_Acc_6M'].mean():.4f}")
print(f"  Min:                {rolling_df['Rolling_Dir_Acc_6M'].min():.4f}")
print(f"  Max:                {rolling_df['Rolling_Dir_Acc_6M'].max():.4f}")
print(f"  Periods below 0.5:  {(rolling_df['Rolling_Dir_Acc_6M'] < 0.5).sum()}")

if rolling_df['Rolling_Dir_Acc_6M'].min() < 0.4:
    print(f"  ⚠ WARNING: Extended weak period detected (rolling accuracy < 0.4)")
if (rolling_df['Rolling_Dir_Acc_6M'] < 0.5).sum() > len(rolling_df) * 0.3:
    print(f"  ⚠ WARNING: Model below 0.5 accuracy for >30% of test period")

print(f"\nRolling directional accuracy (12M window):")
print(f"  Mean:               {rolling_df['Rolling_Dir_Acc_12M'].mean():.4f}")
print(f"  Min:                {rolling_df['Rolling_Dir_Acc_12M'].min():.4f}")
print(f"  Max:                {rolling_df['Rolling_Dir_Acc_12M'].max():.4f}")

print("\nRolling accuracy table (last 10 periods):")
display(rolling_df.tail(10))

spark.createDataFrame(rolling_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_rolling_accuracy"
)
print(f"\n✓ Rolling accuracy saved: {catalog}.{schema}.gold_rolling_accuracy")

# COMMAND ----------

# DBTITLE 1,5. Tail Error Analysis
print("="*60)
print("5. Tail Error Analysis")
print("="*60)

# Tail Error Analysis
# How does model behave on extreme actual returns?
# These are the most important periods for risk management
actual = predictions_df["Actual_3M_Forward_Log_Return"]
p10 = actual.quantile(0.10)
p90 = actual.quantile(0.90)

print(f"\nTail thresholds:")
print(f"  Bottom 10%: returns ≤ {p10:.6f}")
print(f"  Top 10%:    returns ≥ {p90:.6f}")

tail_df = predictions_df[
    (actual <= p10) | (actual >= p90)
].copy()
tail_df["Tail"] = np.where(
    actual[tail_df.index] <= p10, "Bottom 10%", "Top 10%"
)

tail_summary = (
    tail_df
    .groupby("Tail")
    .agg(
        count=("Actual_3M_Forward_Log_Return", "count"),
        avg_actual=("Actual_3M_Forward_Log_Return", "mean"),
        avg_predicted=("Predicted_3M_Forward_Log_Return", "mean"),
        avg_abs_error=("Absolute_Error", "mean"),
        directional_accuracy=("Direction_Correct", "mean"),
    )
    .reset_index()
)
tail_summary["avg_prediction_bias"] = (
    tail_summary["avg_predicted"] - tail_summary["avg_actual"]
)

print("\nTail error analysis (extreme return periods):")
print("  High directional accuracy in tails = model useful for risk management")
print("  Low directional accuracy in tails = model misses regime turning points\n")
display(tail_summary)

print("\nTail diagnostics:")
for _, row in tail_summary.iterrows():
    tail = row["Tail"]
    dir_acc = row["directional_accuracy"]
    bias = row["avg_prediction_bias"]
    
    status = "✓" if dir_acc > 0.5 else "⚠"
    print(f"{status} {tail:<15} — DirAcc: {dir_acc:.4f}, Bias: {bias:+.6f}, n={int(row['count'])}")
    
    if dir_acc < 0.5 and tail == "Bottom 10%":
        print(f"    ⚠ CRITICAL: Model fails to predict crashes (risk management issue)")
    elif dir_acc < 0.5 and tail == "Top 10%":
        print(f"    WARNING: Model fails to predict rallies (opportunity cost)")
    
    if abs(bias) > 0.05:
        if bias > 0 and tail == "Bottom 10%":
            print(f"    ⚠ CRITICAL: Model under-predicts crash magnitude (dangerous)")
        elif bias < 0 and tail == "Top 10%":
            print(f"    WARNING: Model under-predicts rally magnitude")

spark.createDataFrame(tail_summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_tail_diagnostics"
)
print(f"\n✓ Tail diagnostics saved: {catalog}.{schema}.gold_tail_diagnostics")

# COMMAND ----------

# DBTITLE 1,6. Residual Autocorrelation & Summary
print("="*60)
print("6. Residual Autocorrelation")
print("="*60)

# Residual Autocorrelation
# Significant autocorrelation in residuals = model is missing serial structure
# For monthly data, test lags 1, 3, 6
residuals = predictions_df["Residual"].values

print("\nResidual autocorrelation check:")
print("  |AC| < 0.2 = clean (good) ✓")
print("  |AC| > 0.2 = structure remaining (model missing signal) ⚠\n")

for lag in [1, 3, 6]:
    if len(residuals) > lag:
        ac = pd.Series(residuals).autocorr(lag=lag)
        status = "✓" if abs(ac) < 0.2 else "⚠"
        interpretation = "clean" if abs(ac) < 0.2 else "structure remaining"
        print(f"  {status} Lag {lag:2d}: {ac:+.4f}  ({interpretation})")
        
        if abs(ac) > 0.3:
            print(f"      WARNING: Strong autocorrelation — model missing temporal dynamics")

if all(abs(pd.Series(residuals).autocorr(lag=lag)) < 0.2 for lag in [1, 3, 6] if len(residuals) > lag):
    print("\n✓ All lags clean — model has extracted available signal")
else:
    print("\n⚠ Consider adding: lagged target, momentum features, or AR terms")

# Final Diagnostics Summary
print(f"\n{'='*60}")
print(f"Model Diagnostics Summary — {model_name}")
print(f"{'='*60}")
print(f"Predictions:              {len(predictions_df)}")
print(f"Date range:               {predictions_df['Date'].iloc[0]} → {predictions_df['Date'].iloc[-1]}")
print(f"\nCore Metrics:")
print(f"  Directional accuracy:   {dir_acc:.4f}  {('✓ > 0.5' if dir_acc > 0.5 else '⚠ ≤ 0.5')}")
print(f"  Information coeff:      {ic:.4f} (p={pval:.4f})  {('✓ significant' if pval < 0.05 else '⚠ not significant')}")
print(f"  RMSE:                   {rmse:.6f}")
print(f"  R²:                     {r2:.4f}")
print(f"\nBias Check:")
print(f"  Mean residual:          {predictions_df['Residual'].mean():.6f}")
print(f"  Bias t-stat:            {t_stat:.4f} (p={t_pval:.4f})")
print(f"  Unbiased:               {'Yes ✓' if t_pval > 0.05 else 'No — significant bias detected ⚠'}")
print(f"\nRegime Performance:")
for _, row in regime_error.iterrows():
    regime = row["Actual_Return_Regime"]
    dir_acc_regime = row["directional_accuracy"]
    mae_regime = row["avg_abs_error"]
    status = "✓" if dir_acc_regime > 0.5 else "⚠"
    print(f"  {status} {regime:<10} — DirAcc: {dir_acc_regime:.4f}, MAE: {mae_regime:.6f}, n={int(row['count'])}")
print(f"\nTail Performance:")
for _, row in tail_summary.iterrows():
    tail = row["Tail"]
    dir_acc_tail = row["directional_accuracy"]
    bias_tail = row["avg_prediction_bias"]
    status = "✓" if dir_acc_tail > 0.5 else "⚠"
    print(f"  {status} {tail:<15} — DirAcc: {dir_acc_tail:.4f}, Bias: {bias_tail:+.6f}")
print(f"\nTemporal Stability:")
print(f"  6M rolling mean:        {rolling_df['Rolling_Dir_Acc_6M'].mean():.4f}")
print(f"  6M rolling min:         {rolling_df['Rolling_Dir_Acc_6M'].min():.4f}")
print(f"  Periods < 0.5:          {(rolling_df['Rolling_Dir_Acc_6M'] < 0.5).sum()}")
print(f"{'='*60}")

# Overall Health Assessment
print(f"\n{'='*60}")
print(f"Overall Model Health Assessment")
print(f"{'='*60}")

warnings = []
if dir_acc <= 0.5:
    warnings.append("Directional accuracy ≤ 0.5 (no better than random)")
if pval >= 0.05:
    warnings.append("IC not statistically significant")
if t_pval < 0.05:
    warnings.append("Statistically significant bias detected")
if any(regime_error["directional_accuracy"] < 0.5):
    warnings.append("Regime failure detected (directional accuracy < 0.5)")
if any(tail_summary["directional_accuracy"] < 0.5):
    warnings.append("Tail prediction failure (critical for risk management)")
if (rolling_df['Rolling_Dir_Acc_6M'] < 0.5).sum() > len(rolling_df) * 0.3:
    warnings.append("Extended weak periods (>30% of time below 0.5)")
if any(abs(pd.Series(residuals).autocorr(lag=lag)) > 0.3 for lag in [1, 3, 6] if len(residuals) > lag):
    warnings.append("Strong residual autocorrelation (missing signal)")

if len(warnings) == 0:
    print("✓ Model passes all diagnostic checks")
    print("  Ready for production deployment")
else:
    print(f"⚠ {len(warnings)} warning(s) detected:")
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")
    print("\nRecommended actions:")
    if "Regime failure" in str(warnings):
        print("  • Add regime-specific features or consider regime-switching model")
    if "Tail prediction failure" in str(warnings):
        print("  • Add tail-risk features (VIX, credit spreads, volatility)")
    if "bias detected" in str(warnings):
        print("  • Check feature scaling, target leakage, sample selection")
    if "autocorrelation" in str(warnings):
        print("  • Add lagged target, momentum features, or AR model components")
    if "weak periods" in str(warnings):
        print("  • Investigate feature drift and consider model retraining")

print(f"{'='*60}")
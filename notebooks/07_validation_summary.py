# Databricks notebook source
# DBTITLE 1,Validation Summary Header
# MAGIC %md
# MAGIC # 07 Validation Summary
# MAGIC
# MAGIC ## Overview
# MAGIC End-to-end validation across bronze, silver, and gold layers to ensure data quality, pipeline consistency, and model output sanity. Performs 12+ automated checks covering null handling, feature distributions, train/test regime stability, pipeline row counts, feature engineering completeness, and prediction quality.
# MAGIC
# MAGIC ## Purpose
# MAGIC
# MAGIC **Why validate?**
# MAGIC * **Data quality issues** propagate through pipelines and silently corrupt model outputs
# MAGIC * **Regime shifts** between train and test invalidate model assumptions
# MAGIC * **Pipeline bugs** (e.g., incorrect joins, unexpected nulls) often appear as subtle row count mismatches
# MAGIC * **Feature engineering errors** (e.g., log of negative values, wrong lag counts) break modeling
# MAGIC * **Model outputs** require sanity checks (e.g., RMSE, directional accuracy thresholds)
# MAGIC
# MAGIC **When to run:**
# MAGIC * After every pipeline run (bronze → silver → gold → model training)
# MAGIC * Before model deployment or inference
# MAGIC * When debugging unexpected model performance
# MAGIC * As part of scheduled monitoring
# MAGIC
# MAGIC ## Validation Categories
# MAGIC
# MAGIC ### 1. Gold Feature Table — Core Validation
# MAGIC
# MAGIC **Row count & date coverage**:
# MAGIC * Total observations, date range, unique months
# MAGIC * Ensures data loaded correctly from silver
# MAGIC
# MAGIC **Target variable distribution**:
# MAGIC * Mean, std, percentiles, % positive returns
# MAGIC * Null count (should be 0 after silver cleaning)
# MAGIC * Skewness/kurtosis for outlier detection
# MAGIC
# MAGIC **Feature-level null audit**:
# MAGIC * Check every macro feature column for nulls
# MAGIC * Flag features with > 0 nulls (pipeline bug indicator)
# MAGIC
# MAGIC ### 2. Feature Distribution Sanity Checks
# MAGIC
# MAGIC **Statistical summary** (all macro features):
# MAGIC * Mean, std, min, max, skewness, kurtosis
# MAGIC * Flags: high skew (|skew| > 2), high kurtosis (|kurt| > 7)
# MAGIC * Saved to `gold_validation_feature_distributions`
# MAGIC
# MAGIC **Log-transformed features**:
# MAGIC * `log_BANKNIFTY_Close`, `log_Brent_USD_bbl`, `log_USDINR`, `log_India_System_Liquidity_INR_Trn`
# MAGIC * **Critical check**: All values must be positive (log of negative/zero is undefined)
# MAGIC * Flags: non-positive values or nulls
# MAGIC
# MAGIC **Derived features** (range checks):
# MAGIC * **India_Real_Rate**: Expected [-10, 15] (policy rate minus CPI)
# MAGIC * **India_Term_Spread**: Expected [-5, 10] (10Y yield minus policy rate)
# MAGIC * **India_Rate_Differential**: Expected [-5, 10] (India 10Y minus US 10Y)
# MAGIC * Flags: values outside economically plausible ranges
# MAGIC
# MAGIC ### 3. Train/Test Split Validation
# MAGIC
# MAGIC **Regime diagnostics** (80/20 split):
# MAGIC * Compare train vs test: mean return, volatility, % positive, min/max
# MAGIC * Saved to `gold_validation_regime`
# MAGIC
# MAGIC **Regime shift flags**:
# MAGIC * **Mean difference > 0.02**: Large return shift between periods
# MAGIC * **Vol ratio > 1.5 or < 0.67**: Volatility regime change
# MAGIC * **Direction balance diff > 0.15**: % positive returns differ significantly
# MAGIC
# MAGIC **Why it matters**:
# MAGIC * Models trained on low-vol periods fail in high-vol regimes
# MAGIC * Bull-market models underperform in bear markets
# MAGIC * Large shifts suggest test set is out-of-distribution
# MAGIC
# MAGIC ### 4. Pipeline Consistency Checks
# MAGIC
# MAGIC **Silver → Gold row count**:
# MAGIC * Expected loss: max(slow_lags) rows (e.g., 12 rows for 12-month lag window)
# MAGIC * Actual loss should match expected within ±5 rows
# MAGIC * Large discrepancies suggest unexpected nulls or join errors
# MAGIC
# MAGIC **Feature count reconciliation**:
# MAGIC * Count features from config: base macro + lags + momentum + interactions
# MAGIC * Compare to actual gold table column count
# MAGIC * Mismatch indicates feature engineering bug
# MAGIC
# MAGIC **Feature composition** (from config):
# MAGIC * Base macro features (from `macro_features`)
# MAGIC * Fast lag features (fast_vars × fast_lags)
# MAGIC * Slow lag features (slow_vars × slow_lags)
# MAGIC * Momentum features (momentum_vars × momentum_periods)
# MAGIC * Contemporaneous interactions (interaction_vars)
# MAGIC * Lagged interactions (5 hardcoded terms)
# MAGIC
# MAGIC **LassoCV selection consistency**:
# MAGIC * Verify selected features exist in gold table
# MAGIC * Check obs:feature ratio (should be ≥ 15:1 for relaxed selection)
# MAGIC
# MAGIC ### 5. Model Output Validation
# MAGIC
# MAGIC **Model comparison table** (`gold_model_comparison`):
# MAGIC * At least one model should beat 0.5 directional accuracy (better than random)
# MAGIC * RMSE should be < 1.0 (plausible for log returns)
# MAGIC * Baseline model must be present for comparison
# MAGIC
# MAGIC **Prediction table** (`gold_model_predictions`):
# MAGIC * No null actuals or predictions
# MAGIC * Residual distribution: mean ≈ 0, symmetry check (skewness)
# MAGIC * Flag large residuals (> 3 std) as potential outliers
# MAGIC
# MAGIC **Why RMSE < 1.0 matters**:
# MAGIC * Log returns typically range [-0.3, +0.3] for 3-month horizons
# MAGIC * RMSE > 1.0 suggests model is wildly off or data corruption
# MAGIC
# MAGIC **Why directional accuracy matters**:
# MAGIC * < 0.5 = worse than random guessing (model has no signal)
# MAGIC * 0.5-0.6 = weak but potentially tradeable signal
# MAGIC * > 0.6 = strong directional signal
# MAGIC
# MAGIC ### 6. Validation Summary Table
# MAGIC
# MAGIC **Automated checks** (12+ tests):
# MAGIC * Data quality (nulls, log feature validity)
# MAGIC * Regime stability (mean, vol, direction)
# MAGIC * Pipeline consistency (row counts, feature counts)
# MAGIC * Model sanity (directional accuracy, RMSE, baseline presence)
# MAGIC
# MAGIC **Output**: `gold_validation_summary` table
# MAGIC * Each row = one check
# MAGIC * Status: PASSED or WARNING
# MAGIC * Summary: count of passed vs warned checks
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC Four gold-layer validation tables:
# MAGIC
# MAGIC | Table | Contents | Use Case |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_validation_feature_distributions` | Statistical summary for all features | Distribution analysis, outlier detection |
# MAGIC | `gold_validation_regime` | Train vs test regime comparison | Detect regime shifts, assess generalization |
# MAGIC | `gold_validation_summary` | Pass/warn status for all checks | Automated monitoring, pipeline health |
# MAGIC | Console output | Detailed diagnostics with flags | Interactive debugging, investigation |
# MAGIC
# MAGIC ## Interpretation Guide
# MAGIC
# MAGIC ### PASSED Status
# MAGIC * All checks green → pipeline is healthy, ready for inference
# MAGIC * Data quality is good, no distribution anomalies
# MAGIC * Train/test split is stable, models performed reasonably
# MAGIC
# MAGIC ### WARNING Status
# MAGIC
# MAGIC **Common warnings and what they mean**:
# MAGIC
# MAGIC | Warning | Likely Cause | Action |
# MAGIC | --- | --- | --- |
# MAGIC | `null_in_macro_features` | Silver cleaning failed | Check 02_transform_silver nulls |
# MAGIC | `log_feature_nonpositive` | Negative values in log input | Check liquidity shift, data source |
# MAGIC | `regime_mean_diff` | Market regime changed | Consider regime-specific models |
# MAGIC | `regime_vol_ratio` | Volatility regime changed | Adjust risk management, vol scaling |
# MAGIC | `silver_gold_row_loss` | Unexpected nulls in lags | Check feature engineering logic |
# MAGIC | `feature_count_mismatch` | Config/code out of sync | Rerun 03_feature_engineering |
# MAGIC | `best_directional_accuracy < 0.5` | No signal in features | Feature selection issue, revisit EDA |
# MAGIC | `large_residuals_count` | Outliers or model failure | Investigate outlier dates, regime shifts |
# MAGIC
# MAGIC ### Critical vs Non-Critical Warnings
# MAGIC
# MAGIC **Critical** (block inference):
# MAGIC * Nulls in target or features
# MAGIC * Log of non-positive values
# MAGIC * Feature count mismatch
# MAGIC * Best directional accuracy < 0.5
# MAGIC
# MAGIC **Non-critical** (monitor but OK to proceed):
# MAGIC * Regime warnings (expected in real markets)
# MAGIC * 1-2 large residuals (outliers happen)
# MAGIC * High skew/kurtosis in a few features (common in macro data)
# MAGIC
# MAGIC ## Diagnostic Outputs
# MAGIC
# MAGIC **Printed to console**:
# MAGIC * Row counts, date ranges
# MAGIC * Feature null counts (long format)
# MAGIC * Log feature validation results
# MAGIC * Derived feature range checks
# MAGIC * Train/test regime comparison with flags
# MAGIC * Pipeline consistency diagnostics
# MAGIC * Model output sanity checks
# MAGIC * Final pass/warn summary
# MAGIC
# MAGIC **Key metrics to watch**:
# MAGIC * Null count = 0 across all features
# MAGIC * Regime vol ratio between 0.67 and 1.5
# MAGIC * Row loss = max lag window
# MAGIC * Best directional accuracy > 0.5
# MAGIC * Residual mean ≈ 0

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,1. Gold Feature Core Validation
# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats

print("="*60)
print("1. Gold Feature Table — Core Validation")
print("="*60)

# Row count and date range
core_summary = spark.sql(f"""
SELECT
    COUNT(*)                        AS total_rows,
    MIN(Date)                       AS start_date,
    MAX(Date)                       AS end_date,
    DATEDIFF(MAX(Date), MIN(Date))  AS date_span_days,
    COUNT(DISTINCT DATE_TRUNC('MM', Date)) AS unique_months
FROM {catalog}.{schema}.gold_macro_features
""")
print("\nRow count and date coverage:")
display(core_summary)

# Target variable distribution
target_summary = spark.sql(f"""
SELECT
    COUNT(*)                                    AS total_rows,
    SUM(CASE WHEN {target_col} IS NULL
             THEN 1 ELSE 0 END)                 AS null_count,
    ROUND(AVG({target_col}), 6)                 AS mean,
    ROUND(STDDEV({target_col}), 6)              AS std,
    ROUND(MIN({target_col}), 6)                 AS min,
    ROUND(PERCENTILE({target_col}, 0.05), 6)    AS p5,
    ROUND(PERCENTILE({target_col}, 0.25), 6)    AS p25,
    ROUND(PERCENTILE({target_col}, 0.50), 6)    AS median,
    ROUND(PERCENTILE({target_col}, 0.75), 6)    AS p75,
    ROUND(PERCENTILE({target_col}, 0.95), 6)    AS p95,
    ROUND(MAX({target_col}), 6)                 AS max,
    ROUND(SUM(CASE WHEN {target_col} > 0
                   THEN 1 ELSE 0 END)
          / COUNT(*), 4)                        AS pct_positive
FROM {catalog}.{schema}.gold_macro_features
WHERE {target_col} IS NOT NULL
""")
print(f"\nTarget variable ({target_col}) distribution:")
display(target_summary)

# Null check — target variable
null_target = spark.sql(f"""
SELECT *
FROM {catalog}.{schema}.gold_macro_features
WHERE {target_col} IS NULL
""")
null_count = null_target.count()
print(f"\nRows with null {target_col}: {null_count}")
if null_count > 0:
    print("⚠ WARNING: Null target rows found — these should have been dropped in silver.")
    display(null_target)
else:
    print("✓ PASSED — no null target rows.")

# Null check — all feature columns
print(f"\nNull check across all macro features:")
null_check_sql = ", ".join([
    f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}"
    for c in macro_features
])

null_by_feature = spark.sql(f"""
SELECT {null_check_sql}
FROM {catalog}.{schema}.gold_macro_features
""")
display(null_by_feature)

# Convert to long format for readability
null_df = null_by_feature.toPandas().T.reset_index()
null_df.columns = ["feature", "null_count"]
null_df = null_df[null_df["null_count"] > 0].sort_values("null_count", ascending=False)

if len(null_df) > 0:
    print(f"⚠ WARNING: {len(null_df)} features have nulls:")
    display(null_df)
else:
    print("✓ PASSED — no nulls in any macro feature column.")

# COMMAND ----------

# DBTITLE 1,2. Feature Distribution Sanity Checks
print("="*60)
print("2. Feature Distribution Sanity Checks")
print("="*60)

# Feature distribution summary
gold_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
gold_df = gold_df.sort_values("Date").reset_index(drop=True)

print(f"\nComputing distribution statistics for {len(macro_features)} features...")
dist_rows = []
for feat in macro_features:
    if feat not in gold_df.columns:
        continue
    s = gold_df[feat].dropna()
    skew = float(s.skew())
    kurt = float(s.kurt())
    dist_rows.append({
        "feature":  feat,
        "mean":     round(float(s.mean()), 6),
        "std":      round(float(s.std()), 6),
        "min":      round(float(s.min()), 6),
        "max":      round(float(s.max()), 6),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "null_count": int(gold_df[feat].isna().sum()),
        "high_skew":  abs(skew) > 2,
        "high_kurt":  abs(kurt) > 7,
    })

dist_df = pd.DataFrame(dist_rows).sort_values("skewness", ascending=False)
print("\nFeature distribution summary (sorted by skewness):")
display(dist_df)

spark.createDataFrame(dist_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_validation_feature_distributions"
)
print(f"\n✓ Saved: {catalog}.{schema}.gold_validation_feature_distributions")

# Log-transformed feature sanity
# log_Brent, log_USDINR, log_Liquidity must all be positive after transform
log_features = [f for f in macro_features if f.startswith("log_")]
print(f"\n{'='*60}")
print(f"Log-transformed feature validation")
print(f"{'='*60}")
print(f"Features to check: {log_features}\n")

for feat in log_features:
    n_negative = (gold_df[feat] <= 0).sum()
    n_null     = gold_df[feat].isna().sum()
    status     = "✓ PASSED" if n_negative == 0 and n_null == 0 else "⚠ WARNING"
    print(f"{status} — {feat}: {n_negative} non-positive values, {n_null} nulls")

# Derived feature sanity
# Real Rate, Term Spread, Rate Differential should be within plausible ranges
derived_checks = {
    "India_Real_Rate":         (-10, 15),   # policy rate minus CPI
    "India_Term_Spread":       (-5, 10),    # 10Y minus policy rate
    "India_Rate_Differential": (-5, 10),    # India 10Y minus US 10Y
}

print(f"\n{'='*60}")
print(f"Derived feature range validation")
print(f"{'='*60}")
for feat, (lo, hi) in derived_checks.items():
    if feat not in gold_df.columns:
        print(f"  SKIPPED — {feat} not in gold table")
        continue
    n_outside = ((gold_df[feat] < lo) | (gold_df[feat] > hi)).sum()
    actual_min = round(gold_df[feat].min(), 4)
    actual_max = round(gold_df[feat].max(), 4)
    status     = "✓ PASSED" if n_outside == 0 else "⚠ WARNING"
    print(f"{status} — {feat}:")
    print(f"    Actual range:   [{actual_min}, {actual_max}]")
    print(f"    Expected range: [{lo}, {hi}]")
    print(f"    Outside range:  {n_outside} rows")

# COMMAND ----------

# DBTITLE 1,3. Train/Test Split Validation
print("="*60)
print("3. Train/Test Split Validation")
print("="*60)

# Train vs test regime comparison
split_idx = int(len(gold_df) * 0.80)
y_train   = gold_df[target_col].iloc[:split_idx]
y_test    = gold_df[target_col].iloc[split_idx:]

train_start = gold_df["Date"].iloc[0]
train_end   = gold_df["Date"].iloc[split_idx - 1]
test_start  = gold_df["Date"].iloc[split_idx]
test_end    = gold_df["Date"].iloc[-1]

print(f"\nTrain period: {train_start} → {train_end} ({len(y_train)} months)")
print(f"Test period:  {test_start} → {test_end} ({len(y_test)} months)")

regime_df = pd.DataFrame([
    {
        "period":       "train",
        "rows":         len(y_train),
        "mean_return":  round(float(y_train.mean()), 6),
        "std_return":   round(float(y_train.std()), 6),
        "pct_positive": round(float((y_train > 0).mean()), 4),
        "min_return":   round(float(y_train.min()), 6),
        "max_return":   round(float(y_train.max()), 6),
    },
    {
        "period":       "test",
        "rows":         len(y_test),
        "mean_return":  round(float(y_test.mean()), 6),
        "std_return":   round(float(y_test.std()), 6),
        "pct_positive": round(float((y_test > 0).mean()), 4),
        "min_return":   round(float(y_test.min()), 6),
        "max_return":   round(float(y_test.max()), 6),
    },
])

print("\nTrain vs Test regime comparison:")
display(regime_df)

# Regime shift flags
mean_diff  = abs(y_test.mean() - y_train.mean())
vol_ratio  = y_test.std() / y_train.std()
dir_diff   = abs((y_test > 0).mean() - (y_train > 0).mean())

print(f"\n{'='*60}")
print(f"Regime shift diagnostics")
print(f"{'='*60}")
print(f"Mean return difference:    {mean_diff:.4f} "
      f"{'⚠ WARNING' if mean_diff > 0.02 else '✓ OK'}")
print(f"Vol ratio (test/train):    {vol_ratio:.4f} "
      f"{'⚠ WARNING' if vol_ratio > 1.5 or vol_ratio < 0.67 else '✓ OK'}")
print(f"Direction balance diff:    {dir_diff:.4f} "
      f"{'⚠ WARNING' if dir_diff > 0.15 else '✓ OK'}")

spark.createDataFrame(regime_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_validation_regime"
)
print(f"\n✓ Saved: {catalog}.{schema}.gold_validation_regime")

# COMMAND ----------

# DBTITLE 1,4. Pipeline Consistency Checks
print("="*60)
print("4. Pipeline Consistency Checks")
print("="*60)

# Silver → Gold row count consistency
silver_count = spark.table(f"{catalog}.{schema}.silver_macro_base").count()
gold_count   = spark.table(f"{catalog}.{schema}.gold_macro_features").count()

expected_loss = max(slow_lags)  # lag window
actual_loss = silver_count - gold_count

print(f"\nSilver → Gold row count reconciliation:")
print(f"  Silver row count:           {silver_count}")
print(f"  Gold row count:             {gold_count}")
print(f"  Rows lost in feature eng:   {actual_loss}")
print(f"  Expected loss (max lag={max(slow_lags)}): ~{expected_loss} rows")

if abs(actual_loss - expected_loss) > 5:
    print("  ⚠ WARNING: Row loss is larger than expected from lag window alone.")
    print("             Check for unexpected nulls in feature engineering.")
else:
    print("  ✓ PASSED — row loss consistent with lag window.")

# Feature count consistency — config vs gold table
print(f"\n{'='*60}")
print(f"Feature count reconciliation")
print(f"{'='*60}")

gold_cols     = spark.table(f"{catalog}.{schema}.gold_macro_features").columns
gold_features = [c for c in gold_cols if c not in [
    "Date", "BANKNIFTY_Close", "log_BANKNIFTY_Close", target_col
]]

config_base_count    = len(macro_features)
fast_lag_count       = len(fast_vars) * len(fast_lags)
slow_lag_count       = len(slow_vars) * len(slow_lags)
momentum_count       = len(momentum_vars) * len(momentum_periods)
interaction_count    = len(interaction_vars)
lagged_ix_count      = 5  # hardcoded in feature engineering

expected_feature_count = (
    config_base_count
    + fast_lag_count
    + slow_lag_count
    + momentum_count
    + interaction_count
    + lagged_ix_count
)

print(f"\nExpected features from config:")
print(f"  Base macro features:      {config_base_count}")
print(f"  Fast lag features:        {fast_lag_count} ({len(fast_vars)} vars × {len(fast_lags)} lags)")
print(f"  Slow lag features:        {slow_lag_count} ({len(slow_vars)} vars × {len(slow_lags)} lags)")
print(f"  Momentum features:        {momentum_count} ({len(momentum_vars)} vars × {len(momentum_periods)} periods)")
print(f"  Contemporaneous ix:       {interaction_count}")
print(f"  Lagged interactions:      {lagged_ix_count}")
print(f"  Total expected:           {expected_feature_count}")
print(f"  Total in gold table:      {len(gold_features)}")

if len(gold_features) != expected_feature_count:
    print(f"  ⚠ WARNING: Feature count mismatch — "
          f"expected {expected_feature_count}, found {len(gold_features)}")
else:
    print("  ✓ PASSED — feature count matches config expectations.")

# LassoCV selection consistency
print(f"\n{'='*60}")
print(f"LassoCV selection validation")
print(f"{'='*60}")

lasso_sel_df = spark.table(f"{catalog}.{schema}.gold_lasso_feature_selection").toPandas()
selected     = lasso_sel_df[lasso_sel_df["selection_method"].isin(["strict", "relaxed"])]
strict       = lasso_sel_df[lasso_sel_df["selection_method"] == "strict"]

print(f"\nLassoCV optimal alpha:        {lasso_sel_df['optimal_alpha'].iloc[0]:.6f}")
print(f"Strictly selected features:   {len(strict)}")
print(f"Relaxed selected features:    {len(selected)}")
print(f"Obs:feature ratio (relaxed):  {int(len(gold_df) * 0.80) / max(len(selected), 1):.1f}:1")

# Confirm selected features exist in gold table
missing_selected = [f for f in selected["feature"].tolist() if f not in gold_cols]
if missing_selected:
    print(f"  ⚠ WARNING: Selected features missing from gold table: {missing_selected}")
else:
    print("  ✓ PASSED — all selected features present in gold table.")

# COMMAND ----------

# DBTITLE 1,5. Model Output Validation
print("="*60)
print("5. Model Output Validation")
print("="*60)

# Model comparison sanity
model_comp = spark.table(f"{catalog}.{schema}.gold_model_comparison").toPandas()
print("\nModel comparison table:")
display(model_comp)

# Checks
print(f"\n{'='*60}")
print(f"Model output sanity checks")
print(f"{'='*60}")

# At least one model should beat 0.5 directional accuracy
best_dir_acc = model_comp["directional_accuracy"].max()
print(f"Best directional accuracy: {best_dir_acc:.4f} "
      f"{'✓ above 0.5' if best_dir_acc > 0.5 else '⚠ WARNING: all models below 0.5'}")

# RMSE should be in a plausible range for log returns
max_rmse = model_comp["rmse"].max()
print(f"Max RMSE:                  {max_rmse:.6f} "
      f"{'✓ plausible' if max_rmse < 1.0 else '⚠ WARNING: RMSE unusually large'}")

# Check baseline is present
has_baseline = "baseline_single_feature" in model_comp["model_name"].values
print(f"Baseline present:          {'✓ yes' if has_baseline else '⚠ WARNING: baseline missing'}")

# Prediction output validation
print(f"\n{'='*60}")
print(f"Prediction table validation")
print(f"{'='*60}")

preds_df = spark.table(f"{catalog}.{schema}.gold_model_predictions").toPandas()

print(f"\nPrediction rows:       {len(preds_df)}")
print(f"Null actuals:          {preds_df['Actual_3M_Forward_Log_Return'].isna().sum()}")
print(f"Null predictions:      {preds_df['Predicted_3M_Forward_Log_Return'].isna().sum()}")
print(f"Model used:            {preds_df['model_name'].iloc[0]}")

# Residual distribution
residuals = preds_df["Residual"]
print(f"\nResidual distribution:")
print(f"  Mean:     {residuals.mean():.6f}  (should be near 0)")
print(f"  Std:      {residuals.std():.6f}")
print(f"  Skewness: {residuals.skew():.4f}")

# Check for suspiciously large residuals (> 3 std)
std3_threshold = residuals.std() * 3
large_residuals = preds_df[residuals.abs() > std3_threshold]
print(f"  Residuals > 3 std: {len(large_residuals)} rows "
      f"{'✓ OK' if len(large_residuals) <= 3 else '⚠ WARNING: many large residuals'}")
if len(large_residuals) > 0:
    print("\nLarge residual dates:")
    display(large_residuals[["Date", "Actual_3M_Forward_Log_Return", 
                             "Predicted_3M_Forward_Log_Return", "Residual", "model_name"]])

# COMMAND ----------

# DBTITLE 1,6. Validation Summary Table
print("="*60)
print("6. Validation Summary")
print("="*60)

# Compile and save full validation summary
validation_results = [
    # Data quality
    {"check": "null_target_rows",
     "value": null_count,
     "status": "PASSED" if null_count == 0 else "WARNING"},
    {"check": "null_in_macro_features",
     "value": len(null_df),
     "status": "PASSED" if len(null_df) == 0 else "WARNING"},
    # Log feature sanity
    *[
        {"check": f"log_feature_nonpositive_{feat}",
         "value": int((gold_df[feat] <= 0).sum()),
         "status": "PASSED" if (gold_df[feat] <= 0).sum() == 0 else "WARNING"}
        for feat in log_features
    ],
    # Regime
    {"check": "regime_mean_diff",
     "value": round(mean_diff, 4),
     "status": "PASSED" if mean_diff <= 0.02 else "WARNING"},
    {"check": "regime_vol_ratio",
     "value": round(vol_ratio, 4),
     "status": "PASSED" if 0.67 <= vol_ratio <= 1.5 else "WARNING"},
    {"check": "regime_direction_diff",
     "value": round(dir_diff, 4),
     "status": "PASSED" if dir_diff <= 0.15 else "WARNING"},
    # Pipeline consistency
    {"check": "silver_gold_row_loss",
     "value": silver_count - gold_count,
     "status": "PASSED" if abs((silver_count - gold_count) - expected_loss) <= 5 else "WARNING"},
    {"check": "feature_count_matches_config",
     "value": len(gold_features),
     "status": "PASSED" if len(gold_features) == expected_feature_count else "WARNING"},
    {"check": "selected_features_in_gold",
     "value": len(missing_selected),
     "status": "PASSED" if not missing_selected else "WARNING"},
    # Model outputs
    {"check": "best_directional_accuracy",
     "value": round(best_dir_acc, 4),
     "status": "PASSED" if best_dir_acc > 0.5 else "WARNING"},
    {"check": "baseline_in_comparison",
     "value": int(has_baseline),
     "status": "PASSED" if has_baseline else "WARNING"},
    {"check": "large_residuals_count",
     "value": len(large_residuals),
     "status": "PASSED" if len(large_residuals) <= 3 else "WARNING"},
]

validation_df = pd.DataFrame(validation_results)
passed = (validation_df["status"] == "PASSED").sum()
warned = (validation_df["status"] == "WARNING").sum()

print("\nValidation summary table:")
display(validation_df)

print(f"\n{'='*60}")
print(f"VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"Total checks:  {len(validation_df)}")
print(f"  ✓ PASSED:    {passed}")
print(f"  ⚠ WARNING:   {warned}")

if warned > 0:
    print(f"\nWarnings to review:")
    for _, row in validation_df[validation_df["status"] == "WARNING"].iterrows():
        print(f"  ⚠  {row['check']}: {row['value']}")
else:
    print(f"\n🎉 All validations PASSED — pipeline is healthy!")

print(f"{'='*60}")

spark.createDataFrame(validation_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_validation_summary"
)
print(f"\n✓ Saved: {catalog}.{schema}.gold_validation_summary")
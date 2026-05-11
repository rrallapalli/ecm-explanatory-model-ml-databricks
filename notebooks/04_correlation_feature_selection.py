# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # 04 Correlation & Feature Selection
# MAGIC
# MAGIC ## Overview
# MAGIC Performs correlation-based exploratory analysis and initial feature screening for BANKNIFTY 3-month forward return prediction. Combines Pearson and Spearman correlations to detect both linear and monotonic relationships, identifies optimal lag structures, and flags multicollinearity issues.
# MAGIC
# MAGIC **Important**: This is EDA and initial screening — NOT the final feature selection. Model-based selection (regularization, tree importance) will follow.
# MAGIC
# MAGIC ## Input
# MAGIC * **Source**: `gold_macro_features` table
# MAGIC * **Target**: `Return_3M_Forward_Log` (3-month forward log return)
# MAGIC * **Features**: ~80-100 engineered features (base macro + momentum + lags + interactions)
# MAGIC * **Excluded from analysis**: Date, BANKNIFTY_Close, log_BANKNIFTY_Close, target itself
# MAGIC
# MAGIC ## Analysis Pipeline
# MAGIC
# MAGIC ### 1. Dual Correlation Analysis (Pearson + Spearman)
# MAGIC
# MAGIC **Why both?**
# MAGIC * **Pearson**: Measures linear relationships, sensitive to outliers
# MAGIC * **Spearman**: Rank-based, robust to outliers and captures monotonic non-linear relationships
# MAGIC * **Divergence**: Large differences between Pearson and Spearman suggest non-linearity or outlier influence
# MAGIC
# MAGIC **Computed metrics:**
# MAGIC * `pearson_correlation` - Linear correlation with target
# MAGIC * `spearman_correlation` - Rank correlation with target
# MAGIC * `spearman_pvalue` - Statistical significance (α = 0.05)
# MAGIC * `pearson_spearman_divergence` - Absolute difference between the two correlations
# MAGIC * `significant` - Boolean flag for p < 0.05
# MAGIC
# MAGIC **Output**: `gold_correlation_summary` table
# MAGIC
# MAGIC ### 2. Lag Signal Profile
# MAGIC
# MAGIC **Purpose**: Identify which lag of each base macro feature carries the strongest predictive signal.
# MAGIC
# MAGIC **Method**:
# MAGIC * For each base feature (from config `macro_features`)
# MAGIC * Extract all lagged versions (`feature_lag_1`, `feature_lag_3`, etc.)
# MAGIC * Calculate Spearman correlation with target for each lag
# MAGIC * Rank by absolute correlation strength
# MAGIC
# MAGIC **Use cases**:
# MAGIC * **Transmission lag discovery**: How long does it take for policy rate changes to affect equity markets?
# MAGIC * **Lag selection**: If multiple lags of the same variable exist, which is most predictive?
# MAGIC * **Economic interpretation**: Do fast variables (USDINR, Brent) show immediate signal while slow variables (CPI, rates) show delayed signal?
# MAGIC
# MAGIC **Output**: `gold_lag_signal_profile` table
# MAGIC
# MAGIC ### 3. Pairwise Multicollinearity Detection
# MAGIC
# MAGIC **Purpose**: Identify highly correlated feature pairs that may cause instability in linear models.
# MAGIC
# MAGIC **Method**:
# MAGIC * Compute pairwise Pearson correlation matrix across all features
# MAGIC * Flag pairs with |correlation| ≥ 0.80
# MAGIC * Sort by absolute correlation strength
# MAGIC
# MAGIC **Why it matters**:
# MAGIC * High multicollinearity inflates coefficient variance in linear regression
# MAGIC * Can cause unstable/unreliable coefficient estimates
# MAGIC * May require dropping one feature from correlated pairs
# MAGIC * Less critical for tree-based models but still useful for EDA
# MAGIC
# MAGIC **Threshold**: |r| ≥ 0.80 (configurable)
# MAGIC
# MAGIC **Output**: `gold_high_correlation_pairs` table
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC Three gold-layer tables for downstream analysis:
# MAGIC
# MAGIC | Table | Contents | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_correlation_summary` | Feature-target correlations (Pearson + Spearman) | Initial feature ranking, signal strength |
# MAGIC | `gold_lag_signal_profile` | Lag-specific correlations by base feature | Lag selection, transmission analysis |
# MAGIC | `gold_high_correlation_pairs` | Feature-feature pairs with |r| ≥ 0.80 | Multicollinearity screening |
# MAGIC
# MAGIC ## Key Interpretations
# MAGIC
# MAGIC **Strong signal indicators:**
# MAGIC * High |Spearman| with p < 0.05 suggests robust predictive relationship
# MAGIC * Pearson ≈ Spearman suggests linear relationship
# MAGIC * Pearson << Spearman suggests non-linear monotonic relationship (consider transforms)
# MAGIC
# MAGIC **Red flags:**
# MAGIC * Low |Spearman| despite domain logic (check data quality, outliers)
# MAGIC * Large Pearson-Spearman divergence (investigate outliers or non-monotonicity)
# MAGIC * Many high correlation pairs (may need dimensionality reduction)
# MAGIC
# MAGIC ## Diagnostics Printed
# MAGIC * Total features analyzed
# MAGIC * Count of statistically significant features (p < 0.05)
# MAGIC * Count of high collinearity pairs (|r| ≥ 0.80)
# MAGIC * Top 5 features by absolute Spearman correlation

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# Correlation & Feature Selection
# Pearson and Spearman correlation between macro features and 3M forward log return.
# Includes lag signal discovery, multicollinearity inspection, and high-correlation pair flagging.

import pandas as pd
import numpy as np
from scipy import stats

model_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
model_df = model_df.sort_values("Date").reset_index(drop=True)

# Define feature columns
# Drop non-feature columns — updated to match new pipeline
# No 1M/6M forward returns exist anymore, only target_col
drop_cols = [
    "Date",
    "BANKNIFTY_Close",
    "log_BANKNIFTY_Close",
    target_col,
]
feature_cols = [c for c in model_df.columns if c not in drop_cols]
print(f"Features for correlation analysis: {len(feature_cols)}")

# Pearson correlation with target
pearson_df = (
    model_df[feature_cols + [target_col]]
    .corr(numeric_only=True)[[target_col]]
    .reset_index()
    .rename(columns={"index": "feature", target_col: "pearson_correlation"})
    .query("feature != @target_col")
)
pearson_df["abs_pearson"] = pearson_df["pearson_correlation"].abs()
pearson_df = pearson_df.sort_values("abs_pearson", ascending=False)

# Spearman correlation with target
# Spearman is rank-based — more robust to outliers and non-linearity
# Important for macro data which often has fat tails
spearman_corrs = {}
for feat in feature_cols:
    valid = model_df[[feat, target_col]].dropna()
    rho, pval = stats.spearmanr(valid[feat], valid[target_col])
    spearman_corrs[feat] = {"spearman_correlation": rho, "spearman_pvalue": pval}

spearman_df = (
    pd.DataFrame(spearman_corrs)
    .T
    .reset_index()
    .rename(columns={"index": "feature"})
)
spearman_df["abs_spearman"] = spearman_df["spearman_correlation"].abs()
spearman_df["significant"] = spearman_df["spearman_pvalue"] < 0.05

# Combined correlation summary
corr_df = (
    pearson_df
    .merge(spearman_df, on="feature", how="inner")
    .sort_values("abs_spearman", ascending=False)
)

# Flag where Pearson and Spearman disagree significantly
# Large divergence suggests non-linearity or outlier influence
corr_df["pearson_spearman_divergence"] = (
    corr_df["pearson_correlation"] - corr_df["spearman_correlation"]
).abs()

display(corr_df[[
    "feature",
    "pearson_correlation", "abs_pearson",
    "spearman_correlation", "abs_spearman",
    "spearman_pvalue", "significant",
    "pearson_spearman_divergence"
]])

spark.createDataFrame(corr_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_correlation_summary"
)
print(f"Correlation summary saved: {catalog}.{schema}.gold_correlation_summary")




# COMMAND ----------

# Lag signal profile — which lag of each base feature carries most signal?
# Helps identify transmission lags in macro → BANKNIFTY relationship
base_features = list(macro_features)  # from config — no lags
lag_signal_rows = []

for base in base_features:
    # collect all lagged versions of this base feature
    lag_cols = [c for c in feature_cols if c.startswith(base + "_lag_")]
    for lag_col in lag_cols:
        valid = model_df[[lag_col, target_col]].dropna()
        if len(valid) < 30:
            continue
        rho, pval = stats.spearmanr(valid[lag_col], valid[target_col])
        lag_num = int(lag_col.split("_lag_")[-1])
        lag_signal_rows.append({
            "base_feature": base,
            "lag_col": lag_col,
            "lag_months": lag_num,
            "spearman_correlation": rho,
            "abs_spearman": abs(rho),
            "spearman_pvalue": pval,
            "significant": pval < 0.05,
        })

lag_signal_df = (
    pd.DataFrame(lag_signal_rows)
    .sort_values(["base_feature", "lag_months"])
)
display(lag_signal_df)

spark.createDataFrame(lag_signal_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_lag_signal_profile"
)
print(f"Lag signal profile saved: {catalog}.{schema}.gold_lag_signal_profile")


# COMMAND ----------

# Pairwise multicollinearity — feature vs feature
pairwise_corr = model_df[feature_cols].corr(numeric_only=True)
high_corr_pairs = []

for i, col1 in enumerate(pairwise_corr.columns):
    for j, col2 in enumerate(pairwise_corr.columns):
        if i < j:
            corr_value = pairwise_corr.loc[col1, col2]
            if abs(corr_value) >= 0.80:
                high_corr_pairs.append({
                    "feature_1":        col1,
                    "feature_2":        col2,
                    "correlation":      round(corr_value, 4),
                    "abs_correlation":  round(abs(corr_value), 4),
                })

high_corr_df = (
    pd.DataFrame(high_corr_pairs)
    .sort_values("abs_correlation", ascending=False)
    .reset_index(drop=True)
)

print(f"High correlation pairs (|r| >= 0.80): {len(high_corr_df)}")
display(high_corr_df)

spark.createDataFrame(high_corr_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_high_correlation_pairs"
)
print(f"High correlation pairs saved: {catalog}.{schema}.gold_high_correlation_pairs")

# Summary diagnostics
print(f"Total features analysed:          {len(feature_cols)}")
print(f"Significant Spearman (p<0.05):    {corr_df['significant'].sum()}")
print(f"High collinearity pairs (>=0.80): {len(high_corr_df)}")
print(f"Top 5 features by |Spearman|:")
print(corr_df[["feature", "spearman_correlation", "spearman_pvalue"]].head(5).to_string(index=False))
# Databricks notebook source
# DBTITLE 1,SHAP Explainability Header
# MAGIC %md
# MAGIC # 06 Explainability — SHAP
# MAGIC
# MAGIC ## Overview
# MAGIC Explains model predictions using SHAP (SHapley Additive exPlanations) to decompose feature contributions. Identifies which macro features, lags, and Brent interactions drive BANKNIFTY 3-month forward return predictions, and quantifies the consistency of feature importance across XGBoost and Random Forest.
# MAGIC
# MAGIC ## Input
# MAGIC * **Source**: `gold_macro_features` table
# MAGIC * **Feature selection**: `gold_lasso_feature_selection` (uses exact same features as 05_model_training)
# MAGIC * **Models explained**: XGBoost, Random Forest (Ridge excluded due to poor directional accuracy)
# MAGIC * **Split**: Same 80/20 split as model training
# MAGIC * **Scaling**: StandardScaler (fit on train, transform on test)
# MAGIC
# MAGIC ## Why SHAP?
# MAGIC
# MAGIC **Traditional feature importance limitations**:
# MAGIC * Tree-based importances show predictive power but not direction (positive/negative effect)
# MAGIC * Don't show effect for individual predictions
# MAGIC * Sensitive to correlated features
# MAGIC
# MAGIC **SHAP advantages**:
# MAGIC * **Additive attribution**: Prediction = base_value + sum(SHAP values)
# MAGIC * **Local explanations**: SHAP value for each feature × observation
# MAGIC * **Global importance**: Mean |SHAP| across all observations
# MAGIC * **Direction**: Positive SHAP = increases prediction, negative = decreases
# MAGIC * **Theoretically grounded**: Based on cooperative game theory (Shapley values)
# MAGIC
# MAGIC ## Analysis Components
# MAGIC
# MAGIC ### 1. Model Fitting & Validation
# MAGIC
# MAGIC **Models trained**:
# MAGIC * **XGBoost**: 200 trees, max_depth=2, heavy regularization (L1=0.1, L2=2.0)
# MAGIC * **Random Forest**: 200 trees, max_depth=2, min_samples_leaf=5
# MAGIC * **Ridge excluded**: Directional accuracy 0.31 on test set (worse than random)
# MAGIC
# MAGIC **Hyperparameters**: Exact match to 05_model_training for consistency
# MAGIC
# MAGIC **Validation metric**: Directional accuracy (% correct sign predictions)
# MAGIC
# MAGIC ### 2. SHAP Computation
# MAGIC
# MAGIC **Method**: `TreeExplainer` (exact, fast for tree models)
# MAGIC * XGBoost SHAP values computed on test set
# MAGIC * Random Forest SHAP values computed on test set
# MAGIC
# MAGIC **Output**: SHAP matrix (n_test_observations × n_features)
# MAGIC
# MAGIC ### 3. SHAP Visualizations
# MAGIC
# MAGIC **Summary plots (Beeswarm)**:
# MAGIC * Each dot = one observation
# MAGIC * X-axis = SHAP value (impact on prediction)
# MAGIC * Color = feature value (red=high, blue=low)
# MAGIC * Y-axis = features ranked by mean |SHAP|
# MAGIC * **Interpretation**: Shows both importance AND directionality
# MAGIC
# MAGIC **Summary plots (Bar)**:
# MAGIC * Mean absolute SHAP value per feature
# MAGIC * Pure importance ranking (no directionality)
# MAGIC * Comparable to traditional feature importance
# MAGIC
# MAGIC **Dependence plots** (top 3 features):
# MAGIC * X-axis = feature value
# MAGIC * Y-axis = SHAP value for that feature
# MAGIC * **Interpretation**: Non-linear relationships, interactions, thresholds
# MAGIC
# MAGIC ### 4. Cross-Model Ranking Agreement
# MAGIC
# MAGIC **Purpose**: Distinguish robust signals from model-specific artifacts
# MAGIC
# MAGIC **Method**:
# MAGIC * Rank features by mean |SHAP| for each model
# MAGIC * Compute Spearman rank correlation between XGBoost and RF rankings
# MAGIC
# MAGIC **Interpretation**:
# MAGIC * **High correlation (>0.7)**: Both models agree on which features matter → robust signal
# MAGIC * **Low correlation (<0.5)**: Model-specific artifacts dominate → treat rankings with caution
# MAGIC * **Large rank differences**: Features important in one model, not the other → investigate
# MAGIC
# MAGIC ### 5. Feature Group Analysis
# MAGIC
# MAGIC **Categories**:
# MAGIC * Brent (oil price momentum and lags)
# MAGIC * USDINR (currency lags and momentum)
# MAGIC * Liquidity (system liquidity lags)
# MAGIC * Real_Rate (policy rate minus CPI)
# MAGIC * Term_Spread (10Y yield minus policy rate)
# MAGIC * Rate_Differential (India 10Y - US 10Y)
# MAGIC * GDP, Credit_Growth
# MAGIC * Interactions (lagged cross-terms)
# MAGIC * Other
# MAGIC
# MAGIC **Output**: SHAP share % by category → which macro channels dominate?
# MAGIC
# MAGIC ### 6. Brent Deep Dive
# MAGIC
# MAGIC **Focus**: Brent crude oil features (raw, momentum, lags, interactions)
# MAGIC
# MAGIC **Purpose**: Quantify oil shock transmission to equity markets
# MAGIC
# MAGIC **Metrics**:
# MAGIC * Mean |SHAP| per Brent feature
# MAGIC * Cumulative SHAP % contribution
# MAGIC * Rank within full feature set
# MAGIC
# MAGIC ### 7. Interaction Term Analysis
# MAGIC
# MAGIC **Focus**: Lagged interaction features (e.g., `BrentLag3_x_RealRateLag3`)
# MAGIC
# MAGIC **Purpose**: Validate whether interaction terms add explanatory power
# MAGIC
# MAGIC **Key question**: Do interactions rank high? Or are base features + lags sufficient?
# MAGIC
# MAGIC ### 8. Selection Method Impact
# MAGIC
# MAGIC **Comparison**: Strict vs relaxed LassoCV selection
# MAGIC * **Strict**: Features with coef > 0 at optimal alpha (robust signal)
# MAGIC * **Relaxed**: Top N by Lasso ranking (exploratory)
# MAGIC
# MAGIC **Metrics**:
# MAGIC * Total SHAP contribution by selection method
# MAGIC * Average SHAP per feature
# MAGIC * SHAP share %
# MAGIC
# MAGIC **Interpretation**: Does the strictly selected feature dominate? Or do relaxed features add value?
# MAGIC
# MAGIC ## Outputs
# MAGIC
# MAGIC Six gold-layer tables for downstream reporting:
# MAGIC
# MAGIC | Table | Contents | Use Case |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_shap_feature_importance` | Mean \|SHAP\| per feature (XGBoost + RF) | Feature ranking, cumulative importance |
# MAGIC | `gold_shap_rank_agreement` | XGBoost vs RF ranking comparison | Cross-model consistency check |
# MAGIC | `gold_shap_group_importance` | SHAP share % by macro category | Channel-level interpretation |
# MAGIC | `gold_brent_shap_importance` | Brent-specific feature breakdown | Oil shock transmission analysis |
# MAGIC | `gold_interaction_shap_importance` | Interaction term SHAP contributions | Validate interaction value |
# MAGIC | `gold_shap_selection_impact` | Strict vs relaxed SHAP comparison | Selection method validation |
# MAGIC
# MAGIC ## Key Interpretations
# MAGIC
# MAGIC ### Feature Importance
# MAGIC * **High mean \|SHAP\|** = strong influence on predictions (regardless of direction)
# MAGIC * **Cumulative SHAP %** = top N features explain X% of model variance
# MAGIC * **Rank 1-3** = dominant signals
# MAGIC
# MAGIC ### Directionality (from Beeswarm plots)
# MAGIC * **Red dots high, blue dots low** = positive relationship (feature ↑ → prediction ↑)
# MAGIC * **Red dots low, blue dots high** = negative relationship (feature ↑ → prediction ↓)
# MAGIC * **Mixed vertical spread** = non-linear or interaction effects
# MAGIC
# MAGIC ### Dependence Plots
# MAGIC * **Linear trend** = linear relationship
# MAGIC * **Curve or plateau** = non-linear effect or saturation
# MAGIC * **Color patterns** = interaction with another feature
# MAGIC * **Vertical scatter at same X** = other features also matter at that value
# MAGIC
# MAGIC ### Cross-Model Agreement
# MAGIC * **Spearman > 0.7** = robust feature set (both models see same signals)
# MAGIC * **Spearman < 0.5** = model-dependent (treat rankings as exploratory)
# MAGIC * **Large rank diff** = model-specific artifact (investigate outliers, splits)
# MAGIC
# MAGIC ### Selection Method
# MAGIC * **Strict dominates** = LassoCV correctly identified core signal
# MAGIC * **Relaxed adds value** = additional weak signals improve ensemble performance
# MAGIC * **Relaxed dominates** = optimal alpha was too conservative (consider lower alpha)
# MAGIC
# MAGIC ## Diagnostics Printed
# MAGIC * Feature counts (total, strict, relaxed)
# MAGIC * Optimal alpha from LassoCV
# MAGIC * Model directional accuracies
# MAGIC * Top features by SHAP (XGBoost and RF)
# MAGIC * Cross-model rank correlation (Spearman)
# MAGIC * Brent feature count and contribution
# MAGIC * Interaction feature count and contribution
# MAGIC * SHAP share by selection method

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install shap numpy xgboost --upgrade

# COMMAND ----------

# DBTITLE 1,Data Loading & Feature Selection
# Import Libraries
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load Data and Selected Features
model_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
model_df = model_df.sort_values("Date").reset_index(drop=True)

# Load LassoCV selected features from model training
# This ensures SHAP uses exactly the same feature set as 05_model_training
lasso_selection_df = spark.table(
    f"{catalog}.{schema}.gold_lasso_feature_selection"
).toPandas()

selected_features = (
    lasso_selection_df[
        lasso_selection_df["selection_method"].isin(["strict", "relaxed"])
    ]
    .sort_values("abs_coef", ascending=False)
    ["feature"]
    .tolist()
)

strict_features = lasso_selection_df[
    lasso_selection_df["selection_method"] == "strict"
]["feature"].tolist()

optimal_alpha = lasso_selection_df["optimal_alpha"].iloc[0]

print(f"Features loaded from LassoCV selection: {len(selected_features)}")
print(f"  Strictly selected:  {len(strict_features)} → {strict_features}")
print(f"  Relaxed selection:  {len(selected_features) - len(strict_features)}")
print(f"  Optimal alpha:      {optimal_alpha:.6f}")

# Prepare Features and Split
X = model_df[selected_features]
y = model_df[target_col]

split_idx  = int(len(model_df) * 0.80)
X_train    = X.iloc[:split_idx]
X_test     = X.iloc[split_idx:]
y_train    = y.iloc[:split_idx]
y_test     = y.iloc[split_idx:]

print(f"Train rows:        {len(X_train)}")
print(f"Test rows:         {len(X_test)}")
print(f"Feature count:     {len(selected_features)}")
print(f"Obs:feature ratio: {len(X_train)/len(selected_features):.1f}:1")

# Scale Features
# Must match exactly what was done in 05_model_training
scaler         = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=selected_features,
    index=X_train.index,
)
X_test_scaled  = pd.DataFrame(
    scaler.transform(X_test),
    columns=selected_features,
    index=X_test.index,
)

# COMMAND ----------

# DBTITLE 1,Model Fitting
# Fit Models
# Same hyperparameters as 05_model_training
# Ridge excluded — directional accuracy 0.31, worse than random on test set

print("Fitting XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.001,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_dir_acc = (np.sign(y_test.values) == np.sign(xgb_preds)).mean()
print(f"  XGBoost directional accuracy: {xgb_dir_acc:.4f}")

print("Fitting Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=5,
    random_state=42,
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_dir_acc = (np.sign(y_test.values) == np.sign(rf_preds)).mean()
print(f"  RF directional accuracy:      {rf_dir_acc:.4f}")

# COMMAND ----------

# DBTITLE 1,SHAP Computation & Visualization
# SHAP — XGBoost (TreeExplainer)
print("Computing SHAP values — XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_vals = xgb_explainer.shap_values(X_test_scaled)

# XGBoost SHAP Summary — Beeswarm
print("XGBoost SHAP Summary (Beeswarm) — shows importance AND directionality")
shap.summary_plot(
    xgb_shap_vals,
    X_test_scaled,
    plot_type="dot",
    max_display=12,
    show=True,
    plot_size=(12, 7),
)

# XGBoost SHAP Summary — Bar
print("XGBoost SHAP Summary (Bar) — pure importance ranking")
shap.summary_plot(
    xgb_shap_vals,
    X_test_scaled,
    plot_type="bar",
    max_display=12,
    show=True,
    plot_size=(12, 7),
)

# SHAP — Random Forest (TreeExplainer)
print("Computing SHAP values — Random Forest...")
rf_explainer  = shap.TreeExplainer(rf_model)
rf_shap_vals  = rf_explainer.shap_values(X_test_scaled)

# Random Forest SHAP Summary — Beeswarm
print("Random Forest SHAP Summary (Beeswarm)")
shap.summary_plot(
    rf_shap_vals,
    X_test_scaled,
    plot_type="dot",
    max_display=12,
    show=True,
    plot_size=(12, 7),
)

# Random Forest SHAP Summary — Bar
print("Random Forest SHAP Summary (Bar)")
shap.summary_plot(
    rf_shap_vals,
    X_test_scaled,
    plot_type="bar",
    max_display=12,
    show=True,
    plot_size=(12, 7),
)

# COMMAND ----------

# DBTITLE 1,SHAP Importance Tables
# Build SHAP Importance Tables
def build_shap_importance(shap_values, feature_cols, model_name, strict_features):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({
            "feature":          feature_cols,
            "mean_abs_shap":    mean_abs,
            "model_name":       model_name,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df["cumulative_shap_pct"] = (
        df["mean_abs_shap"].cumsum() / df["mean_abs_shap"].sum() * 100
    ).round(2)
    df["selection_method"] = df["feature"].apply(
        lambda f: "strict" if f in strict_features else "relaxed"
    )
    df["rank"] = df.index + 1
    return df

xgb_shap_df = build_shap_importance(
    xgb_shap_vals, selected_features, "xgboost", strict_features
)
rf_shap_df  = build_shap_importance(
    rf_shap_vals,  selected_features, "random_forest", strict_features
)

print("XGBoost SHAP Importance (Top 10):")
display(xgb_shap_df.head(10))

print("\nRandom Forest SHAP Importance (Top 10):")
display(rf_shap_df.head(10))

# Combined table
shap_importance_df = pd.concat(
    [xgb_shap_df, rf_shap_df], ignore_index=True
)

spark.createDataFrame(shap_importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_shap_feature_importance"
)
print(f"\nSHAP importance saved: {catalog}.{schema}.gold_shap_feature_importance")

# COMMAND ----------

# DBTITLE 1,Cross-Model Ranking Agreement
# Cross-model Feature Ranking Agreement
# Where XGBoost and RF agree on importance — high confidence signal
# Where they disagree — model-specific artefact
xgb_ranks = xgb_shap_df[["feature", "mean_abs_shap", "rank"]].rename(
    columns={"mean_abs_shap": "xgb_shap", "rank": "xgb_rank"}
)
rf_ranks  = rf_shap_df[["feature", "mean_abs_shap", "rank"]].rename(
    columns={"mean_abs_shap": "rf_shap", "rank": "rf_rank"}
)

ranks_df = xgb_ranks.merge(rf_ranks, on="feature")
ranks_df["rank_diff"]       = (ranks_df["xgb_rank"] - ranks_df["rf_rank"]).abs()
ranks_df["selection_method"] = ranks_df["feature"].apply(
    lambda f: "strict" if f in strict_features else "relaxed"
)
ranks_df = ranks_df.sort_values("xgb_rank").reset_index(drop=True)

rank_corr, rank_pval = stats.spearmanr(ranks_df["xgb_rank"], ranks_df["rf_rank"])
print(f"XGBoost vs RF SHAP rank correlation: {rank_corr:.4f} (p={rank_pval:.4f})")
print("High correlation = both models agree on which features matter most\n")

print("Top 10 features with ranking comparison:")
display(ranks_df.head(10))

print(f"\nFeatures with large rank disagreement (|diff| > 5):")
large_diff = ranks_df[ranks_df["rank_diff"] > 5].sort_values("rank_diff", ascending=False)
if len(large_diff) > 0:
    display(large_diff.head(10))
else:
    print("None — models show strong agreement")

spark.createDataFrame(ranks_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_shap_rank_agreement"
)
print(f"\nRank agreement saved: {catalog}.{schema}.gold_shap_rank_agreement")

# COMMAND ----------

# DBTITLE 1,Category & Group Analysis
# Macro Feature Group Summary
def categorise_feature(f):
    if "Brent" in f:        return "Brent"
    if "USDINR" in f:       return "USDINR"
    if "Liquidity" in f:    return "Liquidity"
    if "Real_Rate" in f:    return "Real_Rate"
    if "Term_Spread" in f:  return "Term_Spread"
    if "Rate_Diff" in f:    return "Rate_Differential"
    if "GDP" in f:          return "GDP"
    if "Credit" in f:       return "Credit_Growth"
    if "_x_" in f:          return "Interaction"
    return "Other"

print("SHAP importance by macro category:\n")

for label, shap_df in [("XGBoost", xgb_shap_df), ("Random Forest", rf_shap_df)]:
    shap_df["category"] = shap_df["feature"].apply(categorise_feature)
    group_df = (
        shap_df
        .groupby("category")["mean_abs_shap"]
        .sum()
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    group_df["shap_share_pct"] = (
        group_df["mean_abs_shap"] / group_df["mean_abs_shap"].sum() * 100
    ).round(2)
    group_df["model_name"] = label.lower().replace(" ", "_")
    print(f"{label}:")
    display(group_df)
    print()

# Save XGBoost group summary as primary
xgb_shap_df["category"] = xgb_shap_df["feature"].apply(categorise_feature)
group_shap_df = (
    xgb_shap_df
    .groupby("category")["mean_abs_shap"]
    .sum()
    .reset_index()
    .sort_values("mean_abs_shap", ascending=False)
)
group_shap_df["shap_share_pct"] = (
    group_shap_df["mean_abs_shap"] / group_shap_df["mean_abs_shap"].sum() * 100
).round(2)

spark.createDataFrame(group_shap_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_shap_group_importance"
)
print(f"Group importance saved: {catalog}.{schema}.gold_shap_group_importance")

# COMMAND ----------

# DBTITLE 1,Brent & Interaction Deep Dives
# Brent Deep Dive — XGBoost
brent_shap_df = xgb_shap_df[
    xgb_shap_df["feature"].str.contains("Brent", case=False, regex=False)
].reset_index(drop=True)

print(f"Brent-related features in XGBoost SHAP: {len(brent_shap_df)}\n")

if len(brent_shap_df) > 0:
    display(brent_shap_df)
    spark.createDataFrame(brent_shap_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{catalog}.{schema}.gold_brent_shap_importance"
    )
    print(f"Brent SHAP saved: {catalog}.{schema}.gold_brent_shap_importance")
    
    brent_total_shap = brent_shap_df["mean_abs_shap"].sum()
    all_total_shap = xgb_shap_df["mean_abs_shap"].sum()
    brent_share = (brent_total_shap / all_total_shap * 100)
    print(f"\nBrent features account for {brent_share:.2f}% of total SHAP")
else:
    print("No Brent features in selected set — skipping.")

# Interaction Term SHAP
print(f"\n{'='*60}")
interaction_shap_df = xgb_shap_df[
    xgb_shap_df["feature"].str.contains("_x_", case=False, regex=False)
].reset_index(drop=True)

print(f"Interaction features in XGBoost SHAP: {len(interaction_shap_df)}\n")

if len(interaction_shap_df) > 0:
    display(interaction_shap_df)
    spark.createDataFrame(interaction_shap_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{catalog}.{schema}.gold_interaction_shap_importance"
    )
    print(f"Interaction SHAP saved: {catalog}.{schema}.gold_interaction_shap_importance")
    
    interaction_total_shap = interaction_shap_df["mean_abs_shap"].sum()
    interaction_share = (interaction_total_shap / all_total_shap * 100)
    print(f"\nInteraction features account for {interaction_share:.2f}% of total SHAP")
else:
    print("No interaction features in selected set — skipping.")

# COMMAND ----------

# DBTITLE 1,Dependence Plots
# SHAP Dependence Plots — Top 3 XGBoost Features
top3 = xgb_shap_df["feature"].iloc[:3].tolist()
print(f"Dependence plots for top 3 XGBoost features: {top3}\n")

for feat in top3:
    print(f"Dependence plot: {feat}")
    shap.dependence_plot(
        feat,
        xgb_shap_vals,
        X_test_scaled,
        show=True,
    )

print(f"\n{'='*60}\n")

# SHAP Dependence Plots — Top 3 RF Features
top3_rf = rf_shap_df["feature"].iloc[:3].tolist()
print(f"Dependence plots for top 3 RF features: {top3_rf}\n")

for feat in top3_rf:
    print(f"Dependence plot: {feat}")
    shap.dependence_plot(
        feat,
        rf_shap_vals,
        X_test_scaled,
        show=True,
    )

# COMMAND ----------

# DBTITLE 1,Selection Impact & Summary
# Strict vs Relaxed Selection Impact
selection_impact_df = (
    xgb_shap_df
    .groupby("selection_method")["mean_abs_shap"]
    .agg(["sum", "mean", "count"])
    .reset_index()
    .rename(columns={"sum": "total_shap", "mean": "avg_shap", "count": "n_features"})
)
selection_impact_df["shap_share_pct"] = (
    selection_impact_df["total_shap"] / selection_impact_df["total_shap"].sum() * 100
).round(2)

print("SHAP importance by selection method (XGBoost):")
print("  strict = LassoCV strictly selected (coef > 0 at optimal alpha)")
print("  relaxed = top N by Lasso ranking, not strictly selected\n")
display(selection_impact_df)

spark.createDataFrame(selection_impact_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_shap_selection_impact"
)
print(f"\nSelection impact saved: {catalog}.{schema}.gold_shap_selection_impact")

# Summary
print(f"\n{'='*60}")
print(f"SHAP Analysis Complete")
print(f"{'='*60}")
print(f"Models explained:        XGBoost, Random Forest")
print(f"  Ridge excluded:        directional accuracy 0.31 on test set")
print(f"Features analysed:       {len(selected_features)}")
print(f"  Strictly selected:     {len(strict_features)}")
print(f"  Relaxed selection:     {len(selected_features) - len(strict_features)}")
print(f"\nXGBoost directional acc: {xgb_dir_acc:.4f}")
print(f"RF directional acc:      {rf_dir_acc:.4f}")
print(f"\nTop XGBoost feature:     {xgb_shap_df['feature'].iloc[0]}")
print(f"Top RF feature:          {rf_shap_df['feature'].iloc[0]}")
print(f"\nModel rank agreement:    {rank_corr:.4f} (Spearman)")
print(f"  High = both models agree on feature importance")
print(f"  Low  = model-specific artefacts dominate")

if len(brent_shap_df) > 0:
    print(f"\nBrent features:          {len(brent_shap_df)}")
    print(f"Brent SHAP share:        {brent_share:.2f}%")
else:
    print(f"\nBrent features:          0")

if len(interaction_shap_df) > 0:
    print(f"Interaction features:    {len(interaction_shap_df)}")
    print(f"Interaction SHAP share:  {interaction_share:.2f}%")
else:
    print(f"Interaction features:    0")

print(f"{'='*60}")
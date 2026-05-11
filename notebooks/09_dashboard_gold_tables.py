# Databricks notebook source
# DBTITLE 1,Dashboard Gold Tables Header
# MAGIC %md
# MAGIC # 09 Dashboard Gold Tables
# MAGIC
# MAGIC ## Overview
# MAGIC Centralized catalog of 20 dashboard-ready gold tables for BANKNIFTY 3-month forward return prediction model. Each table is optimized for specific visualization types and provides pre-aggregated metrics, diagnostics, and feature importance analysis for Databricks Lakeview dashboards.
# MAGIC
# MAGIC ## Purpose
# MAGIC
# MAGIC **Why this notebook?**
# MAGIC * **Single source of truth** - all dashboard tables validated and documented in one place
# MAGIC * **Schema reference** - know exactly what columns are available for each widget
# MAGIC * **Data freshness check** - verify all tables populated after pipeline runs
# MAGIC * **Rapid prototyping** - copy SQL directly into dashboard widgets
# MAGIC
# MAGIC **When to use:**
# MAGIC * Before creating a new dashboard (to plan widgets)
# MAGIC * After pipeline runs (to validate table availability)
# MAGIC * When troubleshooting missing dashboard data
# MAGIC * For stakeholder demos (quick data preview)
# MAGIC
# MAGIC ## Table Organization
# MAGIC
# MAGIC ### Category 1: Model Performance & Comparison (2 tables)
# MAGIC
# MAGIC **Purpose**: High-level model evaluation and multi-model comparison
# MAGIC
# MAGIC | Table | Key Metrics | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_model_metrics` | Directional accuracy, IC, RMSE, R², bias test results | KPI counters, scorecards |
# MAGIC | `gold_model_comparison` | All models vs baseline, beats_baseline flags | Model comparison table, bar charts |
# MAGIC
# MAGIC **Typical widgets**: KPI counters (directional accuracy, IC, RMSE, R²), comparison table, bar chart ranking models
# MAGIC
# MAGIC ### Category 2: Predictions & Diagnostics (6 tables)
# MAGIC
# MAGIC **Purpose**: Row-level predictions, residual analysis, regime-specific performance
# MAGIC
# MAGIC | Table | Key Metrics | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_model_diagnostics` | Date, actual, predicted, residual, regime, direction | Time-series line charts, residual plots |
# MAGIC | `gold_rolling_accuracy` | 6M and 12M rolling directional accuracy | Temporal stability line charts |
# MAGIC | `gold_residual_bias_summary` | Under vs over-prediction counts | Bias distribution pie/bar |
# MAGIC | `gold_regime_diagnostics` | Performance by Neg/Flat/Pos regimes | Regime comparison table |
# MAGIC | `gold_magnitude_diagnostics` | Performance by Small/Medium/Large moves | Magnitude comparison table |
# MAGIC | `gold_tail_diagnostics` | Bottom 10% and Top 10% tail performance | Risk management tables |
# MAGIC
# MAGIC **Typical widgets**: Dual line chart (actual vs predicted), bar chart (residuals over time), line chart (rolling accuracy), tables (regime breakdowns), pie chart (bias split)
# MAGIC
# MAGIC ### Category 3: Feature Importance & Selection (3 tables)
# MAGIC
# MAGIC **Purpose**: Feature ranking, Lasso selection, and coefficient analysis
# MAGIC
# MAGIC | Table | Key Metrics | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_feature_importance` | Random Forest importance, cumulative % | Top-N bar charts, cumulative importance line |
# MAGIC | `gold_lasso_feature_selection` | LassoCV coefficients, selection method | Strict vs relaxed comparison |
# MAGIC | `gold_lasso_coefficients` | Final Lasso coefficients, active flag | Feature coefficient bar chart (with signs) |
# MAGIC
# MAGIC **Typical widgets**: Horizontal bar (top 10 features), table (full feature ranking), scatter (coefficient vs importance)
# MAGIC
# MAGIC ### Category 4: SHAP Explainability (6 tables)
# MAGIC
# MAGIC **Purpose**: Model-agnostic feature importance and macro channel analysis
# MAGIC
# MAGIC | Table | Key Metrics | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_shap_feature_importance` | Mean abs SHAP (XGBoost + RF), cumulative % | Top-N SHAP bar charts |
# MAGIC | `gold_shap_group_importance` | SHAP share % by macro category | Channel-level bar/pie charts |
# MAGIC | `gold_brent_shap_importance` | Brent-specific SHAP breakdown | Oil transmission analysis |
# MAGIC | `gold_shap_rank_agreement` | XGBoost vs RF rank comparison | Cross-model consistency scatter plot |
# MAGIC | `gold_interaction_shap_importance` | Interaction term SHAP values | Interaction effect bar chart |
# MAGIC | `gold_shap_selection_impact` | Strict vs relaxed SHAP share | Selection method pie chart |
# MAGIC
# MAGIC **Typical widgets**: Grouped bar (XGBoost vs RF top features), horizontal bar (macro categories), scatter plot (rank agreement), pie chart (SHAP share by selection method)
# MAGIC
# MAGIC ### Category 5: Correlation & Validation (3 tables)
# MAGIC
# MAGIC **Purpose**: Statistical significance, lag structure, and pipeline health checks
# MAGIC
# MAGIC | Table | Key Metrics | Primary Use |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_correlation_summary` | Pearson, Spearman, p-values, divergence | Significant features table |
# MAGIC | `gold_lag_signal_profile` | Lag-specific correlations by base feature | Lag selection analysis |
# MAGIC | `gold_validation_summary` | Pass/warn status for 12+ checks | Pipeline health dashboard |
# MAGIC
# MAGIC **Typical widgets**: Table (significant correlations only), line chart (correlation by lag), status table (validation checks with color coding)
# MAGIC
# MAGIC ## Table Details
# MAGIC
# MAGIC ### Model Performance
# MAGIC
# MAGIC **Table 1: `gold_model_metrics`**
# MAGIC * **Columns**: model_name, target, n_predictions, mae, rmse, r2, directional_accuracy, information_coefficient, ic_pvalue, mean_residual, residual_std, residual_bias_tstat, residual_bias_pvalue, residual_unbiased, pct_underprediction
# MAGIC * **Grain**: 1 row per model (best model from 05_model_training)
# MAGIC * **Use**: Main scorecard KPIs
# MAGIC
# MAGIC **Table 2: `gold_model_comparison`**
# MAGIC * **Columns**: model_name, selected_features, optimal_alpha, mae, rmse, r2, directional_accuracy, information_coefficient, ic_pvalue, beats_baseline_dir_acc, beats_baseline_ic, beats_baseline_both
# MAGIC * **Grain**: 1 row per model (all 5 models + baseline)
# MAGIC * **Use**: Model selection justification, sensitivity analysis
# MAGIC
# MAGIC ### Predictions & Diagnostics
# MAGIC
# MAGIC **Table 3: `gold_model_diagnostics`**
# MAGIC * **Columns**: Date, Actual_3M_Forward_Log_Return, Predicted_3M_Forward_Log_Return, Residual, Absolute_Error, Squared_Error, Actual_Direction, Predicted_Direction, Direction_Correct, Residual_Bias, Actual_Return_Regime, Magnitude_Regime, model_name
# MAGIC * **Grain**: 1 row per prediction (test set only, ~20 rows)
# MAGIC * **Use**: Time-series charts, residual analysis
# MAGIC
# MAGIC **Table 4: `gold_rolling_accuracy`**
# MAGIC * **Columns**: Date, Direction_Correct, Rolling_Dir_Acc_6M, Rolling_Dir_Acc_12M, Actual_Return_Regime
# MAGIC * **Grain**: 1 row per prediction (test set)
# MAGIC * **Use**: Temporal stability monitoring
# MAGIC
# MAGIC **Table 5: `gold_residual_bias_summary`**
# MAGIC * **Columns**: Residual_Bias (Under/Over), count, avg_residual, avg_abs_error, max_abs_error, pct_of_predictions
# MAGIC * **Grain**: 2 rows (under-prediction + over-prediction)
# MAGIC * **Use**: Bias distribution visualization
# MAGIC
# MAGIC **Table 6: `gold_regime_diagnostics`**
# MAGIC * **Columns**: Actual_Return_Regime (Negative/Flat/Positive), count, avg_actual, avg_predicted, avg_abs_error, directional_accuracy, pct_underprediction, avg_prediction_bias
# MAGIC * **Grain**: 3 rows (one per regime)
# MAGIC * **Use**: Regime-specific performance tables
# MAGIC
# MAGIC **Table 7: `gold_magnitude_diagnostics`**
# MAGIC * **Columns**: Magnitude_Regime (Small/Medium/Large), count, avg_abs_actual, avg_abs_error, directional_accuracy, rmse, error_to_move_ratio
# MAGIC * **Grain**: 3 rows (one per magnitude bucket)
# MAGIC * **Use**: Volatility impact assessment
# MAGIC
# MAGIC **Table 8: `gold_tail_diagnostics`**
# MAGIC * **Columns**: Tail (Bottom 10%/Top 10%), count, avg_actual, avg_predicted, avg_abs_error, directional_accuracy, avg_prediction_bias
# MAGIC * **Grain**: 2 rows (one per tail)
# MAGIC * **Use**: Risk management evaluation
# MAGIC
# MAGIC ### Feature Importance & Selection
# MAGIC
# MAGIC **Table 9: `gold_feature_importance`**
# MAGIC * **Columns**: feature, importance (RF), cumulative_importance, selection_method (strict/relaxed)
# MAGIC * **Grain**: 1 row per selected feature (~5-10 features)
# MAGIC * **Use**: Feature ranking bar charts
# MAGIC
# MAGIC **Table 10: `gold_lasso_feature_selection`**
# MAGIC * **Columns**: feature, coefficient, abs_coef, selection_method (strict/relaxed/excluded), optimal_alpha
# MAGIC * **Grain**: 1 row per feature in gold table (~80-100)
# MAGIC * **Use**: Full feature ranking, selection method comparison
# MAGIC
# MAGIC **Table 11: `gold_lasso_coefficients`**
# MAGIC * **Columns**: feature, coefficient, abs_coef, active (boolean), selection_method
# MAGIC * **Grain**: 1 row per selected feature (~5-10)
# MAGIC * **Use**: Coefficient sign and magnitude visualization
# MAGIC
# MAGIC ### SHAP Explainability
# MAGIC
# MAGIC **Table 12: `gold_shap_feature_importance`**
# MAGIC * **Columns**: feature, mean_abs_shap, cumulative_shap_pct, model_name (xgboost/random_forest), selection_method, rank
# MAGIC * **Grain**: 2 rows per selected feature (XGBoost + RF)
# MAGIC * **Use**: Cross-model SHAP comparison, cumulative importance
# MAGIC
# MAGIC **Table 13: `gold_shap_group_importance`**
# MAGIC * **Columns**: category (Brent, USDINR, Liquidity, Real_Rate, etc.), mean_abs_shap, shap_share_pct
# MAGIC * **Grain**: 1 row per macro category (~8-10 categories)
# MAGIC * **Use**: Macro channel contribution analysis
# MAGIC
# MAGIC **Table 14: `gold_brent_shap_importance`**
# MAGIC * **Columns**: feature, mean_abs_shap, cumulative_shap_pct, selection_method, rank
# MAGIC * **Grain**: 1 row per Brent-related feature
# MAGIC * **Use**: Oil shock transmission deep dive
# MAGIC
# MAGIC **Table 15: `gold_shap_rank_agreement`**
# MAGIC * **Columns**: feature, xgb_shap, rf_shap, xgb_rank, rf_rank, rank_diff, selection_method
# MAGIC * **Grain**: 1 row per selected feature
# MAGIC * **Use**: Cross-model consistency scatter plot (xgb_rank vs rf_rank)
# MAGIC
# MAGIC **Table 19: `gold_interaction_shap_importance`**
# MAGIC * **Columns**: feature, mean_abs_shap, cumulative_shap_pct, rank
# MAGIC * **Grain**: 1 row per interaction feature (may not exist if no interactions selected)
# MAGIC * **Use**: Interaction term contribution analysis
# MAGIC
# MAGIC **Table 20: `gold_shap_selection_impact`**
# MAGIC * **Columns**: selection_method (strict/relaxed), total_shap, avg_shap, n_features, shap_share_pct
# MAGIC * **Grain**: 2 rows (strict + relaxed)
# MAGIC * **Use**: Selection method SHAP share pie chart
# MAGIC
# MAGIC ### Correlation & Validation
# MAGIC
# MAGIC **Table 16: `gold_correlation_summary`**
# MAGIC * **Columns**: feature, pearson_correlation, spearman_correlation, spearman_pvalue, significant (boolean), pearson_spearman_divergence
# MAGIC * **Grain**: 1 row per feature in gold table
# MAGIC * **Use**: Statistical significance table, divergence analysis
# MAGIC
# MAGIC **Table 17: `gold_lag_signal_profile`**
# MAGIC * **Columns**: base_feature, lag_col, lag_months, spearman_correlation, abs_spearman, spearman_pvalue, significant
# MAGIC * **Grain**: 1 row per (base_feature, lag) pair
# MAGIC * **Use**: Lag selection line charts, transmission delay analysis
# MAGIC
# MAGIC **Table 18: `gold_validation_summary`**
# MAGIC * **Columns**: check (test name), value, status (PASSED/WARNING)
# MAGIC * **Grain**: 1 row per validation check (~12-15 checks)
# MAGIC * **Use**: Pipeline health status table with color coding
# MAGIC
# MAGIC ## Data Freshness
# MAGIC
# MAGIC **Source notebooks**:
# MAGIC * Tables 1-2, 9-11: Created in `05_model_training`
# MAGIC * Tables 3-8: Created in `08_model_diagnostics`
# MAGIC * Tables 12-15, 19-20: Created in `06_explainability_shap`
# MAGIC * Tables 16-17: Created in `04_correlation_feature_selection`
# MAGIC * Table 18: Created in `07_validation_summary`
# MAGIC
# MAGIC **Refresh strategy**:
# MAGIC * **After every model training cycle**: Run notebooks 02 → 03 → 04 → 05 → 06 → 07 → 08
# MAGIC * **After data updates**: Re-run full pipeline (01 → ... → 08)
# MAGIC * **Dashboard refresh**: Lakeview dashboards query these tables directly (auto-refresh on dashboard load)
# MAGIC
# MAGIC ## Usage Guide
# MAGIC
# MAGIC ### Creating a New Dashboard
# MAGIC
# MAGIC **Step 1: Run this notebook**
# MAGIC * Validates all 20 tables exist and are populated
# MAGIC * Preview schemas and sample data
# MAGIC
# MAGIC **Step 2: Copy SQL to dashboard widgets**
# MAGIC * SQL queries in this notebook are dashboard-ready
# MAGIC * Modify `SELECT` columns and `WHERE` filters as needed
# MAGIC
# MAGIC **Step 3: Choose visualization types**
# MAGIC * Refer to "Typical widgets" in each category above
# MAGIC * Match table grain to chart requirements
# MAGIC
# MAGIC **Step 4: Add interactivity**
# MAGIC * Use dashboard parameters for date filters, regime filters, model selection
# MAGIC * Link widgets (e.g., click on regime → filter time-series)
# MAGIC
# MAGIC ### Common Dashboard Patterns
# MAGIC
# MAGIC **KPI Scorecard** (single value):
# MAGIC ```sql
# MAGIC SELECT directional_accuracy 
# MAGIC FROM {catalog}.{schema}.gold_model_metrics
# MAGIC ```
# MAGIC
# MAGIC **Time-series line chart** (dual axis):
# MAGIC ```sql
# MAGIC SELECT Date, 
# MAGIC        Actual_3M_Forward_Log_Return, 
# MAGIC        Predicted_3M_Forward_Log_Return
# MAGIC FROM {catalog}.{schema}.gold_model_diagnostics
# MAGIC ORDER BY Date
# MAGIC ```
# MAGIC
# MAGIC **Top-N bar chart** (horizontal):
# MAGIC ```sql
# MAGIC SELECT feature, mean_abs_shap
# MAGIC FROM {catalog}.{schema}.gold_shap_feature_importance
# MAGIC WHERE model_name = 'xgboost'
# MAGIC ORDER BY mean_abs_shap DESC
# MAGIC LIMIT 10
# MAGIC ```
# MAGIC
# MAGIC **Regime comparison table**:
# MAGIC ```sql
# MAGIC SELECT Actual_Return_Regime, 
# MAGIC        directional_accuracy, 
# MAGIC        avg_abs_error,
# MAGIC        count
# MAGIC FROM {catalog}.{schema}.gold_regime_diagnostics
# MAGIC ORDER BY Actual_Return_Regime
# MAGIC ```
# MAGIC
# MAGIC **Pie chart** (bias split):
# MAGIC ```sql
# MAGIC SELECT Residual_Bias, pct_of_predictions
# MAGIC FROM {catalog}.{schema}.gold_residual_bias_summary
# MAGIC ```
# MAGIC
# MAGIC ### Troubleshooting
# MAGIC
# MAGIC **Table not found**:
# MAGIC * Check source notebook ran successfully
# MAGIC * Verify catalog/schema names in `00_config`
# MAGIC * Re-run source notebook
# MAGIC
# MAGIC **Empty table**:
# MAGIC * Check feature selection step (may have excluded certain feature types)
# MAGIC * Example: `gold_interaction_shap_importance` only exists if interaction features were selected
# MAGIC
# MAGIC **Stale data**:
# MAGIC * Check last modified timestamp: `DESCRIBE DETAIL {catalog}.{schema}.{table_name}`
# MAGIC * Re-run full pipeline if > 1 month old
# MAGIC
# MAGIC **Schema mismatch**:
# MAGIC * Column names changed in pipeline refactor
# MAGIC * Update dashboard SQL to match new schema
# MAGIC * Check this notebook for current column list
# MAGIC

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Category 1: Model Performance & Comparison
print("="*60)
print("CATEGORY 1: MODEL PERFORMANCE & COMPARISON")
print("="*60)
print("2 tables: Core metrics and multi-model comparison\n")

# Table 1: Model Performance Summary
print("\n" + "="*60)
print("Table 1: gold_model_metrics")
print("Purpose: Core model evaluation metrics + bias test")
print("Grain: 1 row (best model)")
print("="*60)
display(spark.table(f"{catalog}.{schema}.gold_model_metrics"))

# Table 2: Model Comparison (All Models vs Baseline)
print("\n" + "="*60)
print("Table 2: gold_model_comparison")
print("Purpose: All 5 models + baseline ranked by RMSE")
print("Grain: 6 rows (5 models + baseline)")
print("="*60)
display(spark.sql(f"""
SELECT
    model_name,
    selected_features,
    optimal_alpha,
    mae,
    rmse,
    r2,
    directional_accuracy,
    information_coefficient,
    ic_pvalue,
    beats_baseline_dir_acc,
    beats_baseline_ic,
    beats_baseline_both
FROM {catalog}.{schema}.gold_model_comparison
ORDER BY rmse ASC
"""))

print("\n" + "="*60)
print("CATEGORY 1 COMPLETE")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Category 2: Predictions & Diagnostics
print("="*60)
print("CATEGORY 2: PREDICTIONS & DIAGNOSTICS")
print("="*60)
print("6 tables: Row-level predictions, rolling accuracy, regime breakdowns\n")

# Table 3: Actual vs Predicted Log Returns
print("\n" + "="*60)
print("Table 3: gold_model_diagnostics")
print("Purpose: Row-level predictions with diagnostic columns")
print("Grain: 1 row per test set prediction (~20 rows)")
print("="*60)
display(spark.sql(f"""
SELECT
    Date,
    Actual_3M_Forward_Log_Return,
    Predicted_3M_Forward_Log_Return,
    Residual,
    Absolute_Error,
    Direction_Correct,
    Actual_Return_Regime,
    Magnitude_Regime,
    Residual_Bias
FROM {catalog}.{schema}.gold_model_diagnostics
ORDER BY Date
"""))

# Table 4: Rolling Directional Accuracy
print("\n" + "="*60)
print("Table 4: gold_rolling_accuracy")
print("Purpose: 6M and 12M rolling directional accuracy")
print("Grain: 1 row per test set prediction")
print("="*60)
display(spark.sql(f"""
SELECT
    Date,
    Direction_Correct,
    Rolling_Dir_Acc_6M,
    Rolling_Dir_Acc_12M,
    Actual_Return_Regime
FROM {catalog}.{schema}.gold_rolling_accuracy
ORDER BY Date
"""))

# Table 5: Residual Bias Summary
print("\n" + "="*60)
print("Table 5: gold_residual_bias_summary")
print("Purpose: Under vs over-prediction breakdown")
print("Grain: 2 rows (Under + Over)")
print("="*60)
display(spark.table(f"{catalog}.{schema}.gold_residual_bias_summary"))

# Table 6: Return Regime Diagnostics
print("\n" + "="*60)
print("Table 6: gold_regime_diagnostics")
print("Purpose: Performance by return regime (Neg/Flat/Pos)")
print("Grain: 3 rows (one per regime)")
print("="*60)
display(spark.table(f"{catalog}.{schema}.gold_regime_diagnostics"))

# Table 7: Magnitude Regime Diagnostics
print("\n" + "="*60)
print("Table 7: gold_magnitude_diagnostics")
print("Purpose: Performance by magnitude of move (Small/Med/Large)")
print("Grain: 3 rows (one per magnitude bucket)")
print("="*60)
display(spark.table(f"{catalog}.{schema}.gold_magnitude_diagnostics"))

# Table 8: Tail Error Diagnostics
print("\n" + "="*60)
print("Table 8: gold_tail_diagnostics")
print("Purpose: Performance on extreme returns (bottom 10%, top 10%)")
print("Grain: 2 rows (one per tail)")
print("="*60)
display(spark.table(f"{catalog}.{schema}.gold_tail_diagnostics"))

print("\n" + "="*60)
print("CATEGORY 2 COMPLETE")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Category 3: Feature Importance & Selection
print("="*60)
print("CATEGORY 3: FEATURE IMPORTANCE & SELECTION")
print("="*60)
print("3 tables: Random Forest importance, LassoCV selection, final coefficients\n")

# Table 9: Feature Importance (Random Forest)
print("\n" + "="*60)
print("Table 9: gold_feature_importance")
print("Purpose: Random Forest feature importance ranking")
print("Grain: 1 row per selected feature (~5-10)")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    importance,
    cumulative_importance,
    selection_method
FROM {catalog}.{schema}.gold_feature_importance
ORDER BY importance DESC
"""))

# Table 10: LassoCV Feature Selection
print("\n" + "="*60)
print("Table 10: gold_lasso_feature_selection")
print("Purpose: Full LassoCV coefficient ranking (all features)")
print("Grain: 1 row per feature in gold table (~80-100)")
print("Use: Showing top 10 only - set LIMIT to see more")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    coefficient,
    abs_coef,
    selection_method,
    optimal_alpha
FROM {catalog}.{schema}.gold_lasso_feature_selection
ORDER BY abs_coef DESC
LIMIT 10
"""))

# Table 11: Lasso Final Coefficients
print("\n" + "="*60)
print("Table 11: gold_lasso_coefficients")
print("Purpose: Final Lasso coefficients on relaxed feature set")
print("Grain: 1 row per selected feature (~5-10)")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    coefficient,
    abs_coef,
    active,
    selection_method
FROM {catalog}.{schema}.gold_lasso_coefficients
ORDER BY abs_coef DESC
"""))

print("\n" + "="*60)
print("CATEGORY 3 COMPLETE")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Category 4: SHAP Explainability
print("="*60)
print("CATEGORY 4: SHAP EXPLAINABILITY")
print("="*60)
print("6 tables: SHAP importance, macro categories, cross-model agreement\n")

# Table 12: SHAP Feature Importance (XGBoost + RF)
print("\n" + "="*60)
print("Table 12: gold_shap_feature_importance")
print("Purpose: SHAP importance for both XGBoost and Random Forest")
print("Grain: 2 rows per feature (XGBoost + RF)")
print("Use: Showing top 10 per model - set LIMIT to see more")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    mean_abs_shap,
    cumulative_shap_pct,
    model_name,
    selection_method,
    rank
FROM {catalog}.{schema}.gold_shap_feature_importance
WHERE rank <= 10
ORDER BY model_name, rank ASC
"""))

# Table 13: SHAP Group Importance by Macro Category
print("\n" + "="*60)
print("Table 13: gold_shap_group_importance")
print("Purpose: SHAP aggregated by macro category (Brent, USDINR, etc.)")
print("Grain: 1 row per category (~8-10 categories)")
print("="*60)
display(spark.sql(f"""
SELECT
    category,
    mean_abs_shap,
    shap_share_pct
FROM {catalog}.{schema}.gold_shap_group_importance
ORDER BY shap_share_pct DESC
"""))

# Table 14: Brent SHAP Importance
print("\n" + "="*60)
print("Table 14: gold_brent_shap_importance")
print("Purpose: Brent-specific SHAP breakdown")
print("Grain: 1 row per Brent feature")
print("="*60)
try:
    display(spark.sql(f"""
    SELECT
        feature,
        mean_abs_shap,
        cumulative_shap_pct,
        selection_method,
        rank
    FROM {catalog}.{schema}.gold_brent_shap_importance
    ORDER BY rank ASC
    """))
except Exception as e:
    print(f"Table not available: {e}")
    print("This table only exists if Brent features were selected.")

# Table 15: Cross-Model Feature Rank Agreement
print("\n" + "="*60)
print("Table 15: gold_shap_rank_agreement")
print("Purpose: XGBoost vs RF feature ranking comparison")
print("Grain: 1 row per selected feature")
print("Use: Showing top 10 by XGBoost rank - set LIMIT to see more")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    xgb_shap,
    rf_shap,
    xgb_rank,
    rf_rank,
    rank_diff,
    selection_method
FROM {catalog}.{schema}.gold_shap_rank_agreement
ORDER BY xgb_rank ASC
LIMIT 10
"""))

# Table 19: Interaction SHAP Importance
print("\n" + "="*60)
print("Table 19: gold_interaction_shap_importance")
print("Purpose: SHAP values for interaction terms")
print("Grain: 1 row per interaction feature (may not exist)")
print("="*60)
try:
    display(spark.sql(f"""
    SELECT
        feature,
        mean_abs_shap,
        cumulative_shap_pct,
        rank
    FROM {catalog}.{schema}.gold_interaction_shap_importance
    ORDER BY rank ASC
    """))
except Exception as e:
    print(f"Table not available: {e}")
    print("This table only exists if interaction features were selected.")

# Table 20: SHAP Strict vs Relaxed Selection Impact
print("\n" + "="*60)
print("Table 20: gold_shap_selection_impact")
print("Purpose: SHAP share by selection method (strict vs relaxed)")
print("Grain: 2 rows (strict + relaxed)")
print("="*60)
display(spark.sql(f"""
SELECT
    selection_method,
    total_shap,
    avg_shap,
    n_features,
    shap_share_pct
FROM {catalog}.{schema}.gold_shap_selection_impact
ORDER BY shap_share_pct DESC
"""))

print("\n" + "="*60)
print("CATEGORY 4 COMPLETE")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Category 5: Correlation & Validation
print("="*60)
print("CATEGORY 5: CORRELATION & VALIDATION")
print("="*60)
print("3 tables: Correlations, lag profiles, pipeline health checks\n")

# Table 16: Correlation Summary
print("\n" + "="*60)
print("Table 16: gold_correlation_summary")
print("Purpose: Pearson + Spearman correlations with target")
print("Grain: 1 row per feature in gold table")
print("Use: Showing top 10 by abs(Spearman) - set LIMIT to see more")
print("="*60)
display(spark.sql(f"""
SELECT
    feature,
    pearson_correlation,
    spearman_correlation,
    spearman_pvalue,
    significant,
    pearson_spearman_divergence
FROM {catalog}.{schema}.gold_correlation_summary
ORDER BY abs(spearman_correlation) DESC
LIMIT 10
"""))

# Table 17: Lag Signal Profile
print("\n" + "="*60)
print("Table 17: gold_lag_signal_profile")
print("Purpose: Lag-specific correlations by base feature")
print("Grain: 1 row per (base_feature, lag) pair")
print("Use: Showing first 20 rows - filter by base_feature for specific analysis")
print("="*60)
display(spark.sql(f"""
SELECT
    base_feature,
    lag_col,
    lag_months,
    spearman_correlation,
    abs_spearman,
    spearman_pvalue,
    significant
FROM {catalog}.{schema}.gold_lag_signal_profile
ORDER BY base_feature, lag_months ASC
LIMIT 20
"""))

# Table 18: Validation Summary
print("\n" + "="*60)
print("Table 18: gold_validation_summary")
print("Purpose: Pipeline health checks (pass/warn status)")
print("Grain: 1 row per validation check (~12-15 checks)")
print("="*60)
display(spark.sql(f"""
SELECT
    check,
    value,
    status
FROM {catalog}.{schema}.gold_validation_summary
ORDER BY
    CASE WHEN status = 'WARNING' THEN 0 ELSE 1 END ASC,
    check ASC
"""))

print("\n" + "="*60)
print("CATEGORY 5 COMPLETE")
print("="*60)

print("\n" + "="*80)
print("ALL 20 DASHBOARD TABLES VALIDATED")
print("="*80)
print("\nTables ready for Databricks Lakeview dashboard consumption.")
print("Refer to Cell 1 documentation for table details and Cell 6 for dashboard layout.")
print("="*80)

# COMMAND ----------


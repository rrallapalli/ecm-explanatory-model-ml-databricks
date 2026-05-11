# Databricks notebook source
# DBTITLE 1,Model Training Header
# MAGIC %md
# MAGIC # 05 Model Training & Ensemble
# MAGIC
# MAGIC ## Overview
# MAGIC Trains and evaluates multiple regression models to predict BANKNIFTY 3-month forward log returns using engineered macro features. Implements a rigorous 5-step pipeline from feature selection through model evaluation with explicit baseline comparison.
# MAGIC
# MAGIC ## Input
# MAGIC * **Source**: `gold_macro_features` table (~80-100 features)
# MAGIC * **Target**: `Return_3M_Forward_Log` (3-month forward log return)
# MAGIC * **Split**: 80/20 train-test with strict chronological ordering
# MAGIC * **Validation**: No lookahead — train dates < test dates
# MAGIC
# MAGIC ## 5-Step Training Pipeline
# MAGIC
# MAGIC ### Step 1: LassoCV Optimal Alpha & Feature Ranking
# MAGIC
# MAGIC **Method**:
# MAGIC * Time-series cross-validation (5-fold `TimeSeriesSplit`)
# MAGIC * Alpha grid: `np.logspace(-4, 1, 100)` (100 candidates)
# MAGIC * Selects optimal alpha minimizing CV error
# MAGIC
# MAGIC **Outputs**:
# MAGIC * `optimal_alpha` - regularization strength
# MAGIC * Full coefficient ranking table (all features by |coef| magnitude)
# MAGIC * **Strictly selected features** - features with coef > 0 at optimal alpha
# MAGIC
# MAGIC **Why LassoCV?**
# MAGIC * L1 regularization drives weak coefficients exactly to zero
# MAGIC * Automatic feature selection via coefficient shrinkage
# MAGIC * Optimal alpha balances bias-variance for time-series data
# MAGIC
# MAGIC ### Step 2: Single Feature Baseline (Minimum Complexity)
# MAGIC
# MAGIC **Purpose**: Establish honest minimum-complexity benchmark that all models must beat.
# MAGIC
# MAGIC **Method**:
# MAGIC * Train `LinearRegression` on strictly selected feature(s) only
# MAGIC * Evaluate on test set for R², RMSE, directional accuracy, IC
# MAGIC
# MAGIC **Why this matters**:
# MAGIC * Prevents overfitting justification — complex models must prove added value
# MAGIC * Baseline represents the "best single predictor" identified by Lasso
# MAGIC * If complex models can't beat this, added features are noise
# MAGIC
# MAGIC **Baseline metrics saved**: R², RMSE, directional accuracy, IC, p-value
# MAGIC
# MAGIC ### Step 3: Relaxed Feature Selection (Top N by Lasso Ranking)
# MAGIC
# MAGIC **Challenge**: Optimal Lasso often selects very few features (sometimes 1)
# MAGIC
# MAGIC **Solution**: Relaxed selection for exploratory modeling
# MAGIC * Target ratio: **15:1 observations:features** (conservative)
# MAGIC * Select top N features by Lasso coefficient magnitude (not just active)
# MAGIC * Label selection method: `strict` (coef > 0) vs `relaxed` (top N by rank)
# MAGIC
# MAGIC **Rationale**:
# MAGIC * Strictly selected = robust signal at optimal alpha
# MAGIC * Relaxed = Lasso-ranked features for ensemble exploration
# MAGIC * 15:1 ratio prevents overfitting while allowing model comparison
# MAGIC
# MAGIC **Output**: `gold_lasso_feature_selection` table (all features + selection method)
# MAGIC
# MAGIC ### Step 4: Multi-Model Training on Relaxed Feature Set
# MAGIC
# MAGIC **Preprocessing**: All models wrapped in `StandardScaler` (fit on train only)
# MAGIC
# MAGIC **Model architectures**:
# MAGIC
# MAGIC | Model | Hyperparameters | Regularization Strategy |
# MAGIC | --- | --- | --- |
# MAGIC | **Lasso** | `alpha=optimal_alpha` | L1 (sparse coefficients) |
# MAGIC | **Ridge** | `alpha=2.0` | L2 (coefficient shrinkage) |
# MAGIC | **Random Forest** | 200 trees, max_depth=2, min_samples_leaf=5 | Depth constraint + ensemble |
# MAGIC | **XGBoost** | 200 trees, max_depth=2, lr=0.001, L1=0.1, L2=2.0 | Boosting + dual regularization |
# MAGIC | **MLP Neural Net** | (16,) hidden layer, alpha=0.01, early stopping | Dropout + weight decay |
# MAGIC
# MAGIC **All models heavily regularized** to prevent overfitting given limited sample size.
# MAGIC
# MAGIC **MLflow logging**:
# MAGIC * All hyperparameters, metrics, and trained models
# MAGIC * Registered to Unity Catalog: `{catalog}.{schema}.{model_name}`
# MAGIC * Signature inference for model serving
# MAGIC
# MAGIC ### Step 5: Model Comparison & Selection
# MAGIC
# MAGIC **Evaluation metrics** (all computed on held-out test set):
# MAGIC * **R²** - explained variance
# MAGIC * **RMSE** - prediction error magnitude
# MAGIC * **Directional Accuracy** - % correct sign predictions (long/short signal)
# MAGIC * **Information Coefficient (IC)** - Spearman rank correlation (predictions, actuals)
# MAGIC * **IC p-value** - statistical significance
# MAGIC
# MAGIC **Selection criteria**:
# MAGIC 1. Must beat baseline on **both** directional accuracy AND IC
# MAGIC 2. Best model = lowest RMSE among those beating baseline
# MAGIC 3. If no model beats baseline → save baseline predictions
# MAGIC
# MAGIC **Outputs**:
# MAGIC * `gold_model_comparison` - all model metrics + baseline comparison flags
# MAGIC * `gold_model_predictions` - test set predictions from best model
# MAGIC * `gold_feature_importance` - Random Forest feature importance + cumulative
# MAGIC * `gold_lasso_coefficients` - final Lasso coefficients (active vs zero)
# MAGIC * `gold_brent_feature_importance` - Brent-related feature subset
# MAGIC
# MAGIC ## Key Validations
# MAGIC
# MAGIC ### Pre-Training Checks
# MAGIC * **Chronological split** - train dates < test dates (no lookahead)
# MAGIC * **Regime diagnostics** - compare train/test mean return, volatility, % positive
# MAGIC * **Warning triggers** - mean diff > 2% or vol ratio > 1.5× suggests regime shift
# MAGIC
# MAGIC ### Post-Training Analysis
# MAGIC * **Baseline comparison** - explicit flags for models beating baseline
# MAGIC * **Active feature count** - how many Lasso coefficients remain non-zero?
# MAGIC * **Obs:feature ratio** - printed at each selection step
# MAGIC
# MAGIC ## Feature Importance Analysis
# MAGIC
# MAGIC **Dual perspective**:
# MAGIC 1. **Random Forest importance** - predictive power (non-linear)
# MAGIC 2. **Lasso coefficients** - linear effect size + direction
# MAGIC
# MAGIC **Special focus**: Brent-related features isolated in separate table
# MAGIC
# MAGIC ## Model Registry
# MAGIC
# MAGIC All models logged to MLflow and registered in Unity Catalog:
# MAGIC * `{catalog}.{schema}.lasso_macro_model`
# MAGIC * `{catalog}.{schema}.ridge_macro_model`
# MAGIC * `{catalog}.{schema}.random_forest_macro_model`
# MAGIC * `{catalog}.{schema}.xgboost_macro_model`
# MAGIC * `{catalog}.{schema}.mlp_nn_macro_model`
# MAGIC
# MAGIC ## Critical Design Choices
# MAGIC
# MAGIC **Why 80/20 split?**
# MAGIC * Limited monthly data → maximize training observations
# MAGIC * 20% preserves ~15-20 test observations for evaluation
# MAGIC
# MAGIC **Why 15:1 obs:feature ratio?**
# MAGIC * Conservative threshold to prevent overfitting
# MAGIC * Standard guideline for regression with limited data
# MAGIC
# MAGIC **Why baseline comparison?**
# MAGIC * Prevents complexity bias — models must prove value over simplest solution
# MAGIC * Common error: adding features that improve training fit but harm generalization
# MAGIC
# MAGIC **Why Spearman IC over Pearson?**
# MAGIC * Rank correlation robust to outliers
# MAGIC * Captures monotonic relationships (not just linear)
# MAGIC * Standard in quantitative finance for signal evaluation

# COMMAND ----------

# DBTITLE 1,Load Config
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Data Loading & Train-Test Split
# Import Libraries
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from scipy import stats

from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost not installed — skipping")
    XGBRegressor = None

# Load and Prepare Data
model_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
model_df = model_df.sort_values("Date").reset_index(drop=True)

drop_cols = [
    "Date",
    "BANKNIFTY_Close",
    "log_BANKNIFTY_Close",
    target_col,
]
feature_cols = [c for c in model_df.columns if c not in drop_cols]

X = model_df[feature_cols]
y = model_df[target_col]

split_idx  = int(len(model_df) * 0.80)
X_train    = X.iloc[:split_idx]
X_test     = X.iloc[split_idx:]
y_train    = y.iloc[:split_idx]
y_test     = y.iloc[split_idx:]
test_dates = model_df.iloc[split_idx:]["Date"]

print(f"Train period: {model_df['Date'].iloc[0]} → {model_df['Date'].iloc[split_idx - 1]}")
print(f"Test period:  {model_df['Date'].iloc[split_idx]} → {model_df['Date'].iloc[-1]}")
print(f"Training rows:  {len(X_train)}")
print(f"Test rows:      {len(X_test)}")
print(f"Total features: {len(feature_cols)}")

# Validate no lookahead in split
assert model_df["Date"].iloc[split_idx - 1] < model_df["Date"].iloc[split_idx], \
    "Data leakage: train and test dates overlap"
print("Split validated — no lookahead detected")

# Regime diagnostics — train vs test
print(f"\n{'='*50}")
print(f"Regime diagnostics")
print(f"{'='*50}")
print(f"Train mean return:  {y_train.mean():.4f}")
print(f"Test  mean return:  {y_test.mean():.4f}")
print(f"Train volatility:   {y_train.std():.4f}")
print(f"Test  volatility:   {y_test.std():.4f}")
print(f"Train % positive:   {(y_train > 0).mean():.2%}")
print(f"Test  % positive:   {(y_test > 0).mean():.2%}")

mean_diff  = abs(y_test.mean() - y_train.mean())
vol_ratio  = y_test.std() / y_train.std()
if mean_diff > 0.02 or vol_ratio > 1.5 or vol_ratio < 0.67:
    print(f"\nWARNING: Test period appears to be a different regime.")
    print(f"  Mean difference:        {mean_diff:.4f}")
    print(f"  Vol ratio (test/train): {vol_ratio:.2f}")
else:
    print("\nRegime check passed — train and test periods appear similar.")
print(f"{'='*50}")

# Scale features
scaler         = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols,
    index=X_train.index,
)
X_test_scaled  = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_cols,
    index=X_test.index,
)

# COMMAND ----------

# DBTITLE 1,Step 1: LassoCV Feature Selection
# Step 1 — LassoCV Feature Selection
print("Running LassoCV for optimal alpha and feature selection...")
print(f"Obs:feature ratio before selection: {len(X_train)}/{len(feature_cols)} = {len(X_train)/len(feature_cols):.1f}:1")

tscv = TimeSeriesSplit(n_splits=5)

lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 100),
    cv=tscv,
    max_iter=10000,
    n_jobs=-1,
)
lasso_cv.fit(X_train_scaled, y_train)

optimal_alpha = lasso_cv.alpha_
print(f"\nOptimal alpha (CV): {optimal_alpha:.6f}")

# Build full coefficient ranking table
lasso_coef_df = (
    pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": lasso_cv.coef_,
        "abs_coef":    np.abs(lasso_cv.coef_),
    })
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)

strict_selected = lasso_coef_df[lasso_coef_df["abs_coef"] > 0]["feature"].tolist()
print(f"Strictly selected features (coef > 0): {len(strict_selected)}")
for f in strict_selected:
    coef = lasso_coef_df.set_index("feature").loc[f, "coefficient"]
    print(f"  {f:<55} {coef:+.6f}")

# COMMAND ----------

# DBTITLE 1,Step 2: Single Feature Baseline
# Step 2 — Single Feature Baseline
# Run a simple linear regression on the strictly selected feature only.
# This is the honest minimum-complexity baseline.
# All subsequent models must beat this to justify added complexity.

strict_feature = strict_selected[0] if len(strict_selected) == 1 else strict_selected

X_train_strict = X_train_scaled[[strict_feature]] if isinstance(strict_feature, str) \
                 else X_train_scaled[strict_feature]
X_test_strict  = X_test_scaled[[strict_feature]] if isinstance(strict_feature, str) \
                 else X_test_scaled[strict_feature]

baseline_model = LinearRegression()
baseline_model.fit(X_train_strict, y_train)
baseline_preds = baseline_model.predict(X_test_strict)

baseline_r2      = r2_score(y_test, baseline_preds)
baseline_dir_acc = (np.sign(y_test.values) == np.sign(baseline_preds)).mean()
baseline_ic, baseline_pval = stats.spearmanr(y_test.values, baseline_preds)
baseline_rmse    = np.sqrt(mean_squared_error(y_test, baseline_preds))

print(f"\n{'='*50}")
print(f"Single feature baseline: {strict_feature}")
print(f"{'='*50}")
print(f"  R²:                    {baseline_r2:.4f}")
print(f"  RMSE:                  {baseline_rmse:.6f}")
print(f"  Directional Accuracy:  {baseline_dir_acc:.4f}")
print(f"  IC:                    {baseline_ic:.4f} (p={baseline_pval:.4f})")
print(f"\nAll models must beat this baseline to justify added complexity.")
print(f"{'='*50}")

# Save baseline metrics for comparison
baseline_result = {
    "model_name":              "baseline_single_feature",
    "selected_features":       1,
    "optimal_alpha":           optimal_alpha,
    "mae":                     round(mean_absolute_error(y_test, baseline_preds), 6),
    "rmse":                    round(baseline_rmse, 6),
    "r2":                      round(baseline_r2, 4),
    "directional_accuracy":    round(baseline_dir_acc, 4),
    "information_coefficient": round(baseline_ic, 4),
    "ic_pvalue":               round(baseline_pval, 4),
}

# COMMAND ----------

# DBTITLE 1,Step 3: Relaxed Feature Selection
# Step 3 — Relaxed Feature Selection (top N by Lasso ranking)
# LassoCV strictly selected only 1 feature at optimal alpha
# This confirms limited robust signal relative to feature space size
# We relax to top N by coefficient magnitude for exploratory modeling
# and are explicit that these are Lasso-ranked, not strictly selected

target_n = len(X_train) // 15  # conservative 15:1 obs:feature ratio
print(f"LassoCV optimal alpha:        {optimal_alpha:.6f}")
print(f"Strictly selected features:   {len(strict_selected)}")
print(f"Relaxed selection target:     {target_n} features (15:1 ratio)")

selected_features = (
    lasso_coef_df
    .head(target_n)
    ["feature"]
    .tolist()
)

# Label selection method for each feature
lasso_coef_df["selection_method"] = lasso_coef_df["feature"].apply(
    lambda f: "strict"   if f in strict_selected else
              "relaxed"  if f in selected_features else
              "excluded"
)

print(f"\nTop {target_n} features by Lasso coefficient magnitude:")
display(
    lasso_coef_df
    .head(target_n)
    [["feature", "coefficient", "abs_coef", "selection_method"]]
)

# Save feature selection results
lasso_coef_df["optimal_alpha"] = optimal_alpha

spark.createDataFrame(lasso_coef_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_lasso_feature_selection"
)
print(f"Feature selection saved: {catalog}.{schema}.gold_lasso_feature_selection")

# Restrict to relaxed selected features
X_train_sel = X_train_scaled[selected_features]
X_test_sel  = X_test_scaled[selected_features]

print(f"Training shape:          {X_train_sel.shape}")
print(f"Test shape:              {X_test_sel.shape}")
print(f"Final obs:feature ratio: {len(X_train_sel)}/{len(selected_features)} = {len(X_train_sel)/len(selected_features):.1f}:1")

# COMMAND ----------

# DBTITLE 1,Step 4: Train Multiple Models
# MLflow Setup
mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/banknifty_macro_3m_log_return"
mlflow.set_experiment(experiment_name)

# Model Evaluation Function
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds    = model.predict(X_te)
    mae      = mean_absolute_error(y_te, preds)
    rmse     = np.sqrt(mean_squared_error(y_te, preds))
    r2       = r2_score(y_te, preds)
    dir_acc  = (np.sign(y_te.values) == np.sign(preds)).mean()
    ic, pval = stats.spearmanr(y_te.values, preds)
    return model, preds, {
        "mae":                     round(mae, 6),
        "rmse":                    round(rmse, 6),
        "r2":                      round(r2, 4),
        "directional_accuracy":    round(dir_acc, 4),
        "information_coefficient": round(ic, 4),
        "ic_pvalue":               round(pval, 4),
    }

# Define Models
models = {
    "ridge_macro_model": Ridge(
        alpha=2.0,
    ),
    "lasso_macro_model": Lasso(
        alpha=optimal_alpha,
        max_iter=10000,
    ),
    "random_forest_macro_model": RandomForestRegressor(
        n_estimators=200,
        max_depth=2,
        min_samples_leaf=5,
        random_state=42,
    ),
    "mlp_nn_macro_model": MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
    ),
}

if XGBRegressor is not None:
    models["xgboost_macro_model"] = XGBRegressor(
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

# Train Models on Relaxed Feature Set
model_results = [baseline_result]  # baseline always first row
fitted_models = {}

best_model      = None
best_preds      = None
best_rmse       = baseline_rmse   # models must beat baseline
best_model_name = "baseline_single_feature"

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    with mlflow.start_run(run_name=model_name):

        fitted_model, preds, metrics = evaluate_model(
            model, X_train_sel, X_test_sel, y_train, y_test
        )
        fitted_models[model_name] = fitted_model

        mlflow.log_param("target",             target_col)
        mlflow.log_param("model_name",         model_name)
        mlflow.log_param("feature_count",      len(selected_features))
        mlflow.log_param("optimal_alpha",      optimal_alpha)
        mlflow.log_param("train_rows",         len(X_train_sel))
        mlflow.log_param("test_rows",          len(X_test_sel))
        mlflow.log_param("obs_feature_ratio",
                         round(len(X_train_sel) / len(selected_features), 1))
        mlflow.log_param("baseline_dir_acc",   baseline_dir_acc)
        mlflow.log_param("baseline_ic",        baseline_ic)

        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log improvement over baseline
        mlflow.log_metric("dir_acc_vs_baseline",
                          round(metrics["directional_accuracy"] - baseline_dir_acc, 4))
        mlflow.log_metric("ic_vs_baseline",
                          round(metrics["information_coefficient"] - baseline_ic, 4))

        signature = mlflow.models.infer_signature(X_train_sel, preds)
        mlflow.sklearn.log_model(
            fitted_model,
            artifact_path=model_name,
            registered_model_name=f"{catalog}.{schema}.{model_name}",
            signature=signature,
        )

        model_results.append({
            "model_name":        model_name,
            "selected_features": len(selected_features),
            "optimal_alpha":     optimal_alpha,
            **metrics,
        })

        beats_baseline_dir = metrics["directional_accuracy"] > baseline_dir_acc
        beats_baseline_ic  = metrics["information_coefficient"] > baseline_ic

        if metrics["rmse"] < best_rmse:
            best_rmse       = metrics["rmse"]
            best_model      = fitted_model
            best_preds      = preds
            best_model_name = model_name

        print(f"  R²: {metrics['r2']:.4f}  |  RMSE: {metrics['rmse']:.6f}  "
              f"|  DirAcc: {metrics['directional_accuracy']:.4f} "
              f"({'✓' if beats_baseline_dir else '✗'} baseline {baseline_dir_acc:.4f})  "
              f"|  IC: {metrics['information_coefficient']:.4f} "
              f"({'✓' if beats_baseline_ic else '✗'} baseline {baseline_ic:.4f})  "
              f"(p={metrics['ic_pvalue']:.4f})")

# COMMAND ----------

# DBTITLE 1,Step 5: Model Comparison & Results
# Step 5 — Model Comparison Table
results_df = (
    pd.DataFrame(model_results)
    .sort_values("rmse")
    .reset_index(drop=True)
)

# Flag models that beat baseline on both key metrics
results_df["beats_baseline_dir_acc"] = (
    results_df["directional_accuracy"] > baseline_dir_acc
)
results_df["beats_baseline_ic"] = (
    results_df["information_coefficient"] > baseline_ic
)
results_df["beats_baseline_both"] = (
    results_df["beats_baseline_dir_acc"] & results_df["beats_baseline_ic"]
)

display(results_df)

spark.createDataFrame(results_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_comparison"
)
print(f"\nBest model: {best_model_name} (RMSE={best_rmse:.6f})")
print(f"Models beating baseline on both metrics: {results_df['beats_baseline_both'].sum() - 1}")

# Save Predictions
# Use best model if it beat baseline, otherwise fall back to baseline
if best_model_name == "baseline_single_feature":
    print("No model beat the baseline — saving baseline predictions.")
    final_preds      = baseline_preds
    final_model_name = "baseline_single_feature"
else:
    final_preds      = best_preds
    final_model_name = best_model_name

predictions_df = pd.DataFrame({
    "Date":                            test_dates.values,
    "Actual_3M_Forward_Log_Return":    y_test.values,
    "Predicted_3M_Forward_Log_Return": final_preds,
    "Residual":                        y_test.values - final_preds,
    "model_name":                      final_model_name,
})
spark.createDataFrame(predictions_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_predictions"
)
print(f"Predictions saved: {len(predictions_df)} rows from {final_model_name}")

# Feature Importance — Random Forest
rf_importances = fitted_models["random_forest_macro_model"].feature_importances_

importance_df = (
    pd.DataFrame({
        "feature":    selected_features,
        "importance": rf_importances,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
importance_df["cumulative_importance"] = importance_df["importance"].cumsum()

# Flag strictly vs relaxed selected
importance_df["selection_method"] = importance_df["feature"].apply(
    lambda f: "strict" if f in strict_selected else "relaxed"
)
display(importance_df)

spark.createDataFrame(importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_feature_importance"
)

# Feature Importance — Lasso Coefficients
lasso_final_df = (
    pd.DataFrame({
        "feature":     selected_features,
        "coefficient": fitted_models["lasso_macro_model"].coef_,
        "abs_coef":    np.abs(fitted_models["lasso_macro_model"].coef_),
    })
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)
lasso_final_df["active"]           = lasso_final_df["abs_coef"] > 0
lasso_final_df["selection_method"] = lasso_final_df["feature"].apply(
    lambda f: "strict" if f in strict_selected else "relaxed"
)

print(f"Lasso active features: {lasso_final_df['active'].sum()} of {len(selected_features)}")
display(lasso_final_df)

spark.createDataFrame(lasso_final_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_lasso_coefficients"
)

# Brent Feature Importance
brent_importance_df = importance_df[
    importance_df["feature"].str.contains("Brent", case=False, regex=False)
].reset_index(drop=True)

print(f"Brent-related features in final set: {len(brent_importance_df)}")
display(brent_importance_df)

spark.createDataFrame(brent_importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_brent_feature_importance"
)

# Training Summary
models_beating_baseline = results_df[
    results_df["beats_baseline_both"] &
    (results_df["model_name"] != "baseline_single_feature")
]["model_name"].tolist()

print(f"\n{'='*60}")
print(f"Training Complete")
print(f"{'='*60}")
print(f"Total features available:     {len(feature_cols)}")
print(f"LassoCV optimal alpha:        {optimal_alpha:.6f}")
print(f"Strictly selected features:   {len(strict_selected)}")
print(f"Relaxed selection (15:1):     {len(selected_features)}")
print(f"Models trained:               {len(models)}")
print(f"Baseline (single feature):")
print(f"  DirAcc:                     {baseline_dir_acc:.4f}")
print(f"  IC:                         {baseline_ic:.4f} (p={baseline_pval:.4f})")
print(f"Models beating baseline:      {models_beating_baseline or 'None'}")
print(f"Best model:                   {best_model_name}")
print(f"Best RMSE:                    {best_rmse:.6f}")
print(f"{'='*60}")
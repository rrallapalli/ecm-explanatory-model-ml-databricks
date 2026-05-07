# How to Run — BANKNIFTY Macro & Brent Interaction Modeling

This guide explains how to run the Databricks notebooks, create the Unity Catalog tables, log models, and import the dashboard.

---

## 1. Create the Databricks folder structure

In Databricks Workspace, create the following structure:

```text
ecm_projects/
├── dashboards/
├── docs/
└── notebooks/
```

Place the files as follows:

```text
dashboards/
└── Macro & Interaction Explanability.lvdash.json

docs/
├── README.md
└── HOW_TO_RUN.md

notebooks/
├── 00_config.py
├── 01_ingest_bronze.py
├── 02_transform_silver.py
├── 03_feature_engineering_lags_interactions.py
├── 04_correlation_feature_selection.py
├── 05_model_training.py
├── 06_explainability_shap.py
├── 07_validation_summary.py
├── 08_model_diagnostics.py
└── 09_dashboard_gold_tables.py
```

---

## 2. Upload the source CSV

The synthetic data used in this project is available at [Data](data/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes (1).csv)

Download this data file and upload this to workspace/banknifty_macro/volume/filestore/ as shown below:
![Data File Upload](images/data_file_upload_location.jpeg)

The current config expects the CSV at:

```text
/Volumes/workspace/banknifty_macro/filestore/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes (1).csv
```

Update this path in `00_config.py` if your file name or volume path is different:

```python
input_file_path = "/Volumes/workspace/banknifty_macro/filestore/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes (1).csv"
```

---

## 3. Confirm catalog and schema

The current notebooks use:

```python
catalog = "workspace"
schema = "banknifty_macro"
```

The config notebook creates the catalog and schema if they do not already exist:

```python
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
```

Expected table namespace:

```text
workspace.banknifty_macro
```

---

## 4. Run the notebooks in order

Run the notebooks in this exact sequence:

```text
00_config
01_ingest_bronze
02_transform_silver
03_feature_engineering_lags_interactions
04_correlation_feature_selection
05_model_training
06_explainability_shap
07_validation_summary
08_model_diagnostics
09_dashboard_gold_tables
```

Each notebook uses:

```python
%run ./00_config
```

So keep all notebooks in the same Databricks folder unless you update the relative path.

---

## 5. What each notebook does

### 00_config

Defines catalog, schema, input file path, target variable, macro features, excluded fields, lag groups, and Brent interaction variables.

### 01_ingest_bronze

Reads the raw CSV, converts Excel serial dates to proper dates, and writes:

```text
workspace.banknifty_macro.bronze_banknifty_macro
```

### 02_transform_silver

Selects the core macro feature set, casts numeric columns, removes nulls, and writes:

```text
workspace.banknifty_macro.silver_macro_base
```

### 03_feature_engineering_lags_interactions

Creates forward returns, log returns, liquidity transformation, lag features, Brent interaction terms, and writes:

```text
workspace.banknifty_macro.gold_macro_features
```

### 04_correlation_feature_selection

Creates feature-target correlations and high-correlation feature-pair tables:

```text
workspace.banknifty_macro.gold_correlation_summary
workspace.banknifty_macro.gold_high_correlation_pairs
```

### 05_model_training

Trains Ridge, Lasso, Random Forest, MLP, and XGBoost if available. Logs metrics to MLflow and registers models in Unity Catalog.

Outputs:

```text
workspace.banknifty_macro.gold_model_comparison
workspace.banknifty_macro.gold_model_predictions
workspace.banknifty_macro.gold_feature_importance
workspace.banknifty_macro.gold_brent_feature_importance
```

### 06_explainability_shap

Runs SHAP explainability and writes:

```text
workspace.banknifty_macro.gold_shap_feature_importance
workspace.banknifty_macro.gold_brent_shap_importance
```

### 07_validation_summary

Creates basic sanity checks and writes:

```text
workspace.banknifty_macro.gold_validation_summary
```

### 08_model_diagnostics

Creates residual, error, bias, and regime diagnostics:

```text
workspace.banknifty_macro.gold_model_diagnostics
workspace.banknifty_macro.gold_model_metrics
workspace.banknifty_macro.gold_residual_bias_summary
workspace.banknifty_macro.gold_regime_diagnostics
```

### 09_dashboard_gold_tables

Displays dashboard-ready tables and provides the suggested dashboard layout.

---

## 6. MLflow and Unity Catalog notes

The model-training notebook uses:

```python
mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/banknifty_macro_brent_interaction"
mlflow.set_experiment(experiment_name)
```

Registered models are written to Unity Catalog using:

```python
registered_model_name=f"{catalog}.{schema}.{model_name}"
```

So the registered model names will look like:

```text
workspace.banknifty_macro.ridge_macro_lags_interactions
workspace.banknifty_macro.lasso_feature_selection
workspace.banknifty_macro.random_forest_macro_interaction_model
workspace.banknifty_macro.mlp_neural_network
workspace.banknifty_macro.xgboost_macro_interaction_model
```

Only register models that you want to preserve, compare, or serve later. For quick experiments, logging runs without registering every model is usually enough.

---

## 7. Import the dashboard JSON

Use the file:

```text
dashboards/Macro & Interaction Explanability.lvdash.json
```

The dashboard expects these tables to exist:

```text
workspace.banknifty_macro.gold_macro_features
workspace.banknifty_macro.bronze_banknifty_macro
workspace.banknifty_macro.gold_model_metrics
workspace.banknifty_macro.gold_model_diagnostics
workspace.banknifty_macro.gold_feature_importance
workspace.banknifty_macro.gold_brent_feature_importance
workspace.banknifty_macro.gold_residual_bias_summary
workspace.banknifty_macro.gold_regime_diagnostics
workspace.banknifty_macro.gold_correlation_summary
```

It also contains an MLflow comparison dataset using system MLflow tables:

```text
system.mlflow.runs_latest
system.mlflow.run_metrics_history
```

If the dashboard does not show MLflow model comparison, update the experiment filter inside the dashboard query to match your Databricks experiment ID.

---

## 8. Dashboard pages

The included dashboard contains three pages:

```text
Macro Drivers
Best Model
Model Diagnostics
```

Main dashboard sections:

- Macro economic drivers
- Banking sector metrics
- Brent oil and currency trends
- Model performance counters
- Actual vs predicted 3M forward returns
- Residual and absolute-error diagnostics
- Feature importance
- Brent transmission story
- MLflow model comparison
- Feature-target correlations

---

## 9. Validation checklist

After running the full pipeline, confirm that these tables exist:

```sql
SHOW TABLES IN workspace.banknifty_macro;
```

Key expected gold tables:

```text
gold_macro_features
gold_correlation_summary
gold_model_comparison
gold_model_predictions
gold_feature_importance
gold_brent_feature_importance
gold_model_diagnostics
gold_model_metrics
gold_residual_bias_summary
gold_regime_diagnostics
```

Check the main target:

```sql
SELECT
  COUNT(*) AS rows,
  MIN(Date) AS start_date,
  MAX(Date) AS end_date,
  AVG(BANKNIFTY_3M_Forward_Return) AS avg_target
FROM workspace.banknifty_macro.gold_macro_features;
```

---

## 10. Common issues

### File path error

Update `input_file_path` in `00_config.py` to match the actual uploaded CSV path.

### `%run ./00_config` fails

Keep all notebooks in the same folder, or update the `%run` path in each notebook.

### Dashboard tables not found

Run notebooks `01` through `09` first, then import or refresh the dashboard.

### MLflow comparison does not show runs

Update the `experiment_id` filter in the dashboard JSON or dashboard query to match your current MLflow experiment.

### XGBoost is skipped

The training notebook skips XGBoost if the package is not available. This does not stop the rest of the pipeline.

### SHAP package issue

Notebook `06_explainability_shap` installs SHAP using:

```python
%pip install shap numpy --upgrade
```

Restart the Python session if Databricks asks for it after package installation.

---

## 11. Recommended portfolio positioning

Use this project to demonstrate:

- Databricks Lakehouse pipeline design
- Unity Catalog table management
- macroeconomic feature engineering
- lag and interaction modeling
- MLflow experiment tracking and model registration
- explainability using feature importance and SHAP
- model diagnostics and dashboard storytelling


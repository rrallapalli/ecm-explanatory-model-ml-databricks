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

The synthetic data used in this project is available at [Dataset](/data/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes%20(1).csv)

Download this data file and upload this to workspace/banknifty_macro/volume/filestore/ as shown below:

![Data File Upload](/images/data_file_upload_location.jpg)

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

Ingests raw BANKNIFTY macroeconomic data into the bronze Delta table using upsert mode for incremental updates.

### 02_transform_silver

Transforms bronze data into the silver macro base table with engineered features for BANKNIFTY prediction modeling.

### 03_feature_engineering_lags_interactions

Creates lag features, momentum indicators, and interaction terms from the silver macro base table to produce the final gold feature set for BANKNIFTY 3-month forward return prediction.

### 04_correlation_feature_selection

Performs correlation-based exploratory analysis and initial feature screening for BANKNIFTY 3-month forward return prediction. Combines Pearson and Spearman correlations to detect both linear and monotonic relationships, identifies optimal lag structures, and flags multicollinearity issues.

### 05_model_training

Trains and evaluates multiple regression models to predict BANKNIFTY 3-month forward log returns using engineered macro features. Implements a rigorous 5-step pipeline from feature selection through model evaluation with explicit baseline comparison.

![ML Flow Experiments](/images/model_experiments_page.jpg)

### 06_explainability_shap

Explains model predictions using SHAP (SHapley Additive exPlanations) to decompose feature contributions. Identifies which macro features, lags, and Brent interactions drive BANKNIFTY 3-month forward return predictions, and quantifies the consistency of feature importance across XGBoost and Random Forest.

### 07_validation_summary

End-to-end validation across bronze, silver, and gold layers to ensure data quality, pipeline consistency, and model output sanity. Performs 12+ automated checks covering null handling, feature distributions, train/test regime stability, pipeline row counts, feature engineering completeness, and prediction quality.

### 08_model_diagnostics

Comprehensive model diagnostics across 8 dimensions to assess prediction quality, stability, bias, and regime-specific performance. Identifies where the model excels, where it struggles, and whether prediction errors are systematic or random.

### 09_dashboard_gold_tables

Centralized catalog of 20 dashboard-ready gold tables for BANKNIFTY 3-month forward return prediction model. Each table is optimized for specific visualization types and provides pre-aggregated metrics, diagnostics, and feature importance analysis for Databricks Lakeview dashboards.

---

## 6. MLflow and Unity Catalog notes

The model-training notebook uses:

```python
mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/banknifty_macro_3m_log_return"
mlflow.set_experiment(experiment_name)
```

Registered models are written to Unity Catalog using:

```python
registered_model_name=f"{model_name}"
```

So the registered model names will look like:

```text
workspace.banknifty_macro.ridge_macro_model
workspace.banknifty_macro.lasso_macro_model
workspace.banknifty_macro.random_forest_macro_model
workspace.banknifty_macro.mlp_nn_model
workspace.banknifty_macro.xgboost_macro_model
```

Only register models that you want to preserve, compare, or serve later. For quick experiments, logging runs without registering every model is usually enough.

---

## 7. Import the dashboard JSON

Use the file:

```text
dashboards/BANKNIFTY Macro Explanatory Analysis.lvdash.json
```
---

## 8. Dashboard pages

The included dashboard contains three pages:

```text
- Macro economic drivers
- Model Performance
- Macro & Brent Transmission
- Model performance counters
- Model Diagnostics
- Statistical Validity
```

---

## 9. Common issues

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

## 10. Recommended portfolio positioning

Use this project to demonstrate:

- Databricks Lakehouse pipeline design
- Unity Catalog table management
- macroeconomic feature engineering
- lag and interaction modeling
- MLflow experiment tracking and model registration
- explainability using feature importance and SHAP
- model diagnostics and dashboard storytelling


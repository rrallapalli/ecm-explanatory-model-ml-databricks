# ML-driven Explanatory Modeling - BANKNIFTY Macro & Brent Interactions — Databricks Lakehouse Pipeline

This project is an end-to-end Databricks Free Edition / Unity Catalog pipeline for explaining future BANKNIFTY returns using macroeconomic drivers, lag effects, Brent crude oil interactions, model diagnostics, and a Databricks dashboard.

The project is designed as a portfolio-ready analytics and ML case study that answers three questions:

1. Which macro variables are associated with future BANKNIFTY performance?
2. How well can multiple macro variables explain BANKNIFTY forward returns when modeled together?
3. How does Brent crude interact with inflation, FX, policy rates, yields, liquidity, and credit growth?

---
## ML in Databricks - Lakehouse Medallion Architecture 
![Medallion Architecture](/images/ecm_ml_explanatory_model_medallion_architecture.png)

## Project Structure

```text
ecm_projects/
├── dashboards/
│   └── Macro & Interaction Explanability.lvdash.json
├── docs/
│   ├── README.md
│   └── HOW_TO_RUN.md
└── notebooks/
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

## Primary Target Variable

```text
BANKNIFTY_3M_Forward_Return
```

The 3-month forward return is used as the main dependent variable because macro variables, oil prices, yields, policy rates, credit growth, and liquidity usually transmit into banking equity performance with a lag rather than only on the same date.

Additional forward returns are also engineered for analysis:

```text
BANKNIFTY_1M_Forward_Return
BANKNIFTY_3M_Forward_Return
BANKNIFTY_6M_Forward_Return
```

---

## Feature Scope

The final feature set focuses on macro and Brent transmission variables:

```text
Brent_USD_bbl
USDINR
India_CPI_YoY_Pct
India_Policy_Rate_Pct
India_10Y_GSec_Yield_Pct
US10Y_Yield_Pct
India_Real_GDP_YoY_Pct
India_Bank_Credit_Growth_YoY_Pct
India_System_Liquidity_INR_Trn
```

Excluded fields include direct index comparators, valuation ratios, and regime labels that could make the model less interpretable for the macro-transmission objective.

---

## Feature Engineering

The pipeline creates:

- BANKNIFTY forward return targets for 1M, 3M, and 6M horizons
- Brent and USDINR log returns
- shifted log transformation for system liquidity
- fast-variable lags at 1M, 3M, and 6M
- slow-variable lags at 3M, 6M, and 12M
- direct Brent interaction terms
- lagged Brent transmission terms

Example Brent interaction features:

```text
BrentReturn_x_USDINR_Log_Return_1M
BrentReturn_x_India_CPI_YoY_Pct
BrentReturn_x_India_Policy_Rate_Pct
BrentReturn_x_India_10Y_GSec_Yield_Pct
BrentReturn_x_Log_System_Liquidity
BrentReturn_x_India_Bank_Credit_Growth_YoY_Pct
BrentReturn_lag3_x_CPI_lag3
BrentReturn_lag6_x_PolicyRate_lag6
BrentReturn_lag3_x_USDINRReturn_lag1
```

---

## Lakehouse Tables

The pipeline writes the following Delta tables under:

```text
workspace.banknifty_macro
```

Core tables:

```text
bronze_banknifty_macro
silver_macro_base
gold_macro_features
```

Modeling and explainability tables:

```text
gold_correlation_summary
gold_high_correlation_pairs
gold_model_comparison
gold_model_predictions
gold_feature_importance
gold_brent_feature_importance
gold_shap_feature_importance
gold_brent_shap_importance
```

Validation and diagnostics tables:

```text
gold_validation_summary
gold_model_diagnostics
gold_model_metrics
gold_residual_bias_summary
gold_regime_diagnostics
```

---

## Modeling Approach

The training notebook compares multiple regression models using a time-based train/test split:

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor, if available
- MLP Neural Network

Each model is wrapped in a scikit-learn pipeline with `StandardScaler` preprocessing.

Metrics logged:

```text
MAE
RMSE
R²
Directional Accuracy
```

Models are logged with MLflow and registered to Unity Catalog using:

```python
mlflow.set_registry_uri("databricks-uc")
registered_model_name=f"{catalog}.{schema}.{model_name}"
```

---

## Explainability Layer

The explainability notebook uses SHAP with a Random Forest model to identify the strongest macro, lag, and Brent interaction effects.

Outputs include:

```text
gold_shap_feature_importance
gold_brent_shap_importance
```

---

## Model Diagnostics Layer

The diagnostic notebook creates dashboard-ready tables for:

- actual vs predicted 3M forward returns
- residuals and absolute error
- over-prediction vs under-prediction bias
- return-regime performance
- directional accuracy

Return regimes are grouped into:

```text
Negative: below -5%
Flat: -5% to +5%
Positive: above +5%
```

---

## Dashboard

The project includes a Databricks dashboard JSON file:

```text
Macro & Interaction Explanability.lvdash.json
```

The dashboard contains pages for:

1. Macro Drivers
   ![Macro Drivers](/images/macro_drivers_page.jpg)
   
2. Best Model
   ![Best Model](/images/best_model_page.jpg)
   
3. Model Diagnostics
   ![Model Diagnostics](/images/model_diagnostics_page.jpg)
   
It uses dashboard datasets built from the gold tables and includes:

- BANKNIFTY trend
- Brent price trend
- FX, GDP, CPI, policy rate, yield, credit growth, and liquidity charts
- banking-sector metrics
- model performance counters
- actual vs predicted returns
- residual diagnostics
- feature importance
- Brent interaction importance
- MLflow model comparison
- feature-target correlations

---

## Recommended Execution Order

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

Then import the dashboard JSON from the `dashboards/` folder.

[How to Run the Project](docs/how_to_run.md)

---

## Notes

This is an explainability-oriented modeling project, not a trading strategy. The goal is to understand macro relationships, lag effects, Brent crude transmission, and model behavior rather than to produce investment advice. 
The data used is synthetically generated and might have actual values for some variables, depending on the availability of public sources.

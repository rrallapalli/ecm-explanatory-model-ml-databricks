# 📈 ML-driven Explanatory Modeling — BANKNIFTY Macro & Brent Interactions
### Databricks Lakehouse + Macro Modeling + Brent Transmission Analysis

---

## 🚀 What This Project Demonstrates

This project showcases how financial institutions can transform raw macroeconomic and market data into explainable macro-financial intelligence using Databricks Lakehouse, Unity Catalog, MLflow, and AI/BI dashboards.

How macroeconomic variables, lag effects, and Brent crude oil interactions can be modeled to explain future BANKNIFTY returns

Built on Databricks Free Edition + Unity Catalog, the project showcases:

- End-to-end Medallion Architecture (Bronze → Silver → Gold)
- PySpark feature engineering pipelines
- Macro & lag interaction modeling
- Brent crude transmission analysis
- MLflow experiment tracking & model registry
- SHAP explainability
- Model diagnostics & residual analysis
- Databricks AI/BI dashboards

---

## 🎯 Business Problem

Banking indices are highly sensitive to macroeconomic conditions:

- Inflation
- Interest rates
- Bond yields
- Liquidity
- Credit growth
- Currency movements
- Oil price shocks

However, macro transmission effects are rarely instantaneous.

This project answers:

1. Which macro variables are associated with future BANKNIFTY performance?
2. How well can macro variables explain BANKNIFTY forward returns collectively?
3. How does Brent crude oil interact with inflation, FX, rates, and liquidity transmission?

---

## 🧠 Key Differentiator: Brent Transmission Modeling

Instead of treating Brent crude as a standalone variable, the project models:

- Brent × Inflation interactions
- Brent × FX interactions
- Brent × Policy Rate interactions
- Brent × Liquidity interactions
- Lagged Brent transmission effects

This creates a more realistic macro-transmission framework for explanatory modeling.

---

## 🧱 Architecture (Databricks Lakehouse)

![Medallion Architecture](/images/ecm_ml_explanatory_model_medallion_architecture.png)

---

## 📊 Dashboard (Databricks AI/BI)

The dashboard is built directly on Gold Delta tables.

### Dashboard Pages

### 1. Macro Time Series

![Macro Time Series](/images/macro_time_series_page.jpg)

### 2. Model Performance

![Best Model](/images/model_performance_page.jpg)

### 3. Macro & Brent Transmission Page

![Macro & Brent Transmission](/images/macro_brent_transmission_page.jpg)

### 4. Model Diagnostics

![Model Diagnostics](/images/model_diagnostics_page.jpg)

### 5. Statistical Validity

![Statistical Validity](/images/Statistical_validity_page.jpg)

---

## 🧠 Analytical Flow

### 1. Macro Trend Analysis

Analyze:

- BANKNIFTY
- Brent crude
- USDINR
- CPI
- GDP
- Policy rates
- Bond yields
- Credit growth
- Liquidity

---

### 2. Feature Engineering & Lag Modeling

The pipeline creates:

- Forward return targets
- Log returns
- Lag structures
- Brent interaction terms
- Transmission variables

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

### 3. Explanatory Modeling

Models trained:

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor
- MLP Neural Network

Metrics tracked:

```text
MAE
RMSE
R²
Directional Accuracy
```

Models are tracked using MLflow and registered in Unity Catalog.

---

### 4. Explainability & Diagnostics

The project includes:

- SHAP explainability
- Residual diagnostics
- Regime diagnostics
- Directional accuracy analysis
- Feature importance
- Brent-specific importance analysis

---

## 🧱 Lakehouse Architecture Layers

### Bronze Layer

- raw ingestion layer
- Excel serial date conversion
- raw Delta persistence

### Silver Layer

- cleaned and conformed macro dataset
- typed columns
- null handling

### Gold Layer

- feature engineering
- model outputs
- diagnostics
- explainability
- dashboard-ready analytics

---

## 📂 Project Structure

```text
ecm_projects/
├── dashboards/
│   └── BANKNIFTY Macro Explanatory Analysis.lvdash.json
│
├── docs/
│   ├── HOW_TO_RUN.md
│
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

## 🛠️ Tech Stack

- Databricks Lakehouse
- Unity Catalog
- PySpark
- Delta Tables
- MLflow
- SHAP
- scikit-learn
- XGBoost
- Databricks AI/BI Dashboards

---

## ▶️ How to Run

Refer [How To Run](/docs/how_to_run.md)

Recommended notebook execution order:

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

---

## 💡 Key Insight

> This project focuses on explainability-oriented macro modeling rather than pure price prediction — helping understand how macro transmission and Brent interactions influence banking sector performance over time.

---

## ⚠️ Notes

- This is an explanatory analytics project, not investment advice.
- The dataset is synthetic with selected macro variables aligned to public macroeconomic indicators.
- Correlation analysis is used for exploration and screening, not as the only feature selection method.

---

## 📬 Contact

If you're interested in discussing Data, AI, or Databricks solutions, feel free to connect.

Rakesh Rallapalli  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/rakesh-rallapalli/)

# Databricks notebook source
# MAGIC %md
# MAGIC # 09 Dashboard Gold Tables
# MAGIC Validates and displays dashboard-ready gold tables for Databricks dashboards.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 1 — Model Performance Summary

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.gold_model_metrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 2 — Actual vs Predicted Returns

# COMMAND ----------

display(spark.sql(f"""
SELECT
    Date,
    Actual_3M_Forward_Return,
    Predicted_3M_Forward_Return,
    Residual,
    Absolute_Error,
    Direction_Correct
FROM {catalog}.{schema}.gold_model_diagnostics
ORDER BY Date
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 3 — Residual Bias

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.gold_residual_bias_summary"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 4 — Return Regime Diagnostics

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.gold_regime_diagnostics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 5 — Feature Importance

# COMMAND ----------

display(spark.sql(f"""
SELECT *
FROM {catalog}.{schema}.gold_feature_importance
ORDER BY importance DESC
LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Table 6 — Brent Feature Importance

# COMMAND ----------

display(spark.sql(f"""
SELECT *
FROM {catalog}.{schema}.gold_brent_feature_importance
ORDER BY importance DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard Suggested Layout
# MAGIC
# MAGIC **Dashboard Title:** BANKNIFTY Macro & Brent Interaction Model
# MAGIC
# MAGIC **Section 1: Model Performance**
# MAGIC - R²
# MAGIC - RMSE
# MAGIC - MAE
# MAGIC - Directional Accuracy
# MAGIC
# MAGIC **Section 2: Actual vs Predicted Returns**
# MAGIC - Line chart: Actual vs Predicted 3M Forward Return
# MAGIC - Bar chart: Residuals over time
# MAGIC - Line chart: Absolute error over time
# MAGIC
# MAGIC **Section 3: Macro Driver Importance**
# MAGIC - Top 20 feature importance
# MAGIC - Brent feature importance
# MAGIC
# MAGIC **Section 4: Diagnostics**
# MAGIC - Residual bias
# MAGIC - Return regime diagnostics
# MAGIC - Directional accuracy by return regime
# MAGIC
# MAGIC **Section 5: Brent Transmission Story**
# MAGIC - Brent × CPI
# MAGIC - Brent × USDINR
# MAGIC - Brent × Policy Rate
# MAGIC - Brent lag interactions

# COMMAND ----------


# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Validation Summary
# MAGIC Basic data validation and sanity checks for the modeling dataset.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

validation_summary = spark.sql(f"""
SELECT
    COUNT(*) AS total_rows,
    MIN(Date) AS start_date,
    MAX(Date) AS end_date,
    AVG(BANKNIFTY_3M_Forward_Return) AS avg_3m_forward_return,
    STDDEV(BANKNIFTY_3M_Forward_Return) AS std_3m_forward_return,
    MIN(BANKNIFTY_3M_Forward_Return) AS min_3m_forward_return,
    MAX(BANKNIFTY_3M_Forward_Return) AS max_3m_forward_return
FROM {catalog}.{schema}.gold_macro_features
""")

display(validation_summary)

validation_summary.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_validation_summary"
)

# COMMAND ----------

null_check = spark.sql(f"""
SELECT *
FROM {catalog}.{schema}.gold_macro_features
WHERE BANKNIFTY_3M_Forward_Return IS NULL
""")

display(null_check)

# COMMAND ----------


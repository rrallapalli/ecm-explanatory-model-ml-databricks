# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Transform Silver
# MAGIC Selects the final macro feature set, casts columns, removes invalid rows, and creates the silver base table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql.functions import col

bronze_df = spark.table(f"{catalog}.{schema}.bronze_banknifty_macro")

required_cols = ["Date", "BANKNIFTY_Close"] + macro_features

missing_cols = [c for c in required_cols if c not in bronze_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

silver_df = bronze_df.select(*required_cols)

for c in required_cols:
    if c != "Date":
        silver_df = silver_df.withColumn(c, col(c).cast("double"))

silver_df = silver_df.dropna().orderBy("Date")

display(silver_df.limit(10))

# COMMAND ----------

silver_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.silver_macro_base"
)

print(f"Silver table created: {catalog}.{schema}.silver_macro_base")
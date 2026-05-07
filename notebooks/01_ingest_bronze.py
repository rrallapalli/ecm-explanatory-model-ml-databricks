# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest Bronze
# MAGIC Reads the raw BANKNIFTY macro dataset and writes it to a bronze Delta table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Read CSV and convert Excel date format
from pyspark.sql.functions import col, date_add, lit

raw_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_file_path)
)

# Convert Excel serial date to proper date
# Excel serial dates are days since 1899-12-30
raw_df = raw_df.withColumn("Date", date_add(lit("1899-12-30"), col("Date").cast("int")))

display(raw_df.limit(10))

# COMMAND ----------

raw_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.bronze_banknifty_macro"
)

print(f"Bronze table created: {catalog}.{schema}.bronze_banknifty_macro")
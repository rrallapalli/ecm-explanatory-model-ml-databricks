# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest Bronze Layer
# MAGIC
# MAGIC ## Purpose
# MAGIC Ingests raw BANKNIFTY macroeconomic data into the bronze Delta table using **upsert mode** for incremental updates.
# MAGIC
# MAGIC ## Source Data
# MAGIC * **Location:** `/Volumes/workspace/banknifty_macro/source/BANKNIFTY_Macro_Data.csv`
# MAGIC * **Format:** CSV with header, Excel serial date format
# MAGIC * **Key Fields:** Date, BANKNIFTY, Brent oil prices, USD/INR, India macroeconomic indicators, banking sector metrics
# MAGIC
# MAGIC ## Ingestion Strategy
# MAGIC * **Mode:** Upsert (Delta MERGE)
# MAGIC * **Primary Key:** Date
# MAGIC * **Logic:**
# MAGIC   * If table exists → Merge on Date (update existing records, insert new ones)
# MAGIC   * If table doesn't exist → Create table with initial load
# MAGIC * **Date Transformation:** Converts Excel serial dates (days since 1899-12-30) to proper date format
# MAGIC
# MAGIC ## Output Table
# MAGIC * **Name:** `workspace.banknifty_macro.bronze_banknifty_macro`
# MAGIC * **Format:** Delta Lake
# MAGIC * **Update Pattern:** Incremental upserts allow for data corrections and new observations without full refresh

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

# DBTITLE 1,Upsert to bronze table using Delta merge
from delta.tables import DeltaTable

table_name = f"{catalog}.{schema}.bronze_banknifty_macro"

# Check if table exists
if spark.catalog.tableExists(table_name):
    # Perform upsert using Delta merge
    delta_table = DeltaTable.forName(spark, table_name)
    
    delta_table.alias("target").merge(
        source=raw_df.alias("source"),
        condition="target.Date = source.Date"
    ).whenMatchedUpdateAll(
    ).whenNotMatchedInsertAll(
    ).execute()
    
    print(f"Bronze table updated (upsert): {table_name}")
else:
    # Table doesn't exist, create it
    raw_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    print(f"Bronze table created: {table_name}")
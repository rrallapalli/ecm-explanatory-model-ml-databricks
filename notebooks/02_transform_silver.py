# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # 02 Transform Silver
# MAGIC
# MAGIC ## Overview
# MAGIC Transforms bronze data into the silver macro base table with engineered features for BANKNIFTY prediction modeling.
# MAGIC
# MAGIC ## Input
# MAGIC * **Source**: `bronze_banknifty_macro` table
# MAGIC * **Frequency**: Daily observations
# MAGIC * **Columns**: Market indices and macroeconomic indicators
# MAGIC
# MAGIC ## Transformations
# MAGIC
# MAGIC ### 1. Column Selection & Casting
# MAGIC * Validates presence of all required columns from config
# MAGIC * Casts numeric columns to `double` type
# MAGIC
# MAGIC ### 2. Monthly Resampling
# MAGIC * **Target frequency**: Month-end observations
# MAGIC * **BANKNIFTY_Close**: Maximum daily value per month
# MAGIC * **Fast market variables**: Monthly mean (USDINR, Brent oil)
# MAGIC * **Slow macro variables**: Last observation carried forward (policy rates, yields, CPI, liquidity)
# MAGIC
# MAGIC ### 3. Log Transformations
# MAGIC * `log_BANKNIFTY_Close` - log of index close
# MAGIC * `log_Brent_USD_bbl` - log of Brent crude price
# MAGIC * `log_USDINR` - log of USD/INR exchange rate
# MAGIC * `log_India_System_Liquidity_INR_Trn` - log of liquidity (with shift for negative values)
# MAGIC
# MAGIC ### 4. Derived Rate Features
# MAGIC * **India_Real_Rate**: Policy rate minus CPI inflation
# MAGIC * **India_Term_Spread**: 10Y yield minus policy rate
# MAGIC * **India_Rate_Differential**: India 10Y yield minus US 10Y yield
# MAGIC
# MAGIC ### 5. Target Variable
# MAGIC * **Return_3M_Forward_Log**: 3-month forward log return of BANKNIFTY
# MAGIC * Computed as: `log(BANKNIFTY_Close[t+3] / BANKNIFTY_Close[t])`
# MAGIC
# MAGIC ### 6. Data Cleaning
# MAGIC * Null value audit pre/post resampling
# MAGIC * Drop rows with any null values (primarily affects last 3 months due to forward return window)
# MAGIC
# MAGIC ## Output
# MAGIC * **Table**: `silver_macro_base`
# MAGIC * **Frequency**: Monthly observations
# MAGIC * **Write mode**: Overwrite with schema overwrite
# MAGIC * **Metadata**: Liquidity shift value stored as table property for downstream reproducibility
# MAGIC
# MAGIC ## Key Metrics Tracked
# MAGIC * Total rows before/after resampling
# MAGIC * Null counts by column
# MAGIC * Liquidity statistics (min, max, negative counts)
# MAGIC * Rows dropped during cleaning

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql.functions import (
    col, log, lead, last, mean, trunc,
    max as spark_max, count, when, lit
)
from pyspark.sql.window import Window

bronze_df = spark.table(f"{catalog}.{schema}.bronze_banknifty_macro")

# Validate raw columns from config
missing_cols = [c for c in raw_required_cols if c not in bronze_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

silver_df = bronze_df.select(*raw_required_cols)

for c in raw_required_cols:
    if c != "Date":
        silver_df = silver_df.withColumn(c, col(c).cast("double"))

# Resample to month-end
silver_df = silver_df.withColumn("Month", trunc(col("Date"), "MM"))

agg_exprs = (
    [spark_max("BANKNIFTY_Close").alias("BANKNIFTY_Close")]
    + [mean(v).alias(v) for v in fast_market_vars]
    + [last(v, ignorenulls=True).alias(v) for v in slow_macro_vars]
)

monthly_df = (
    silver_df
    .groupBy("Month")
    .agg(*agg_exprs)
    .withColumnRenamed("Month", "Date")
    .orderBy("Date")
)

# Null audit post-resampling
total_rows = monthly_df.count()
print(f"Total rows after resampling: {total_rows}")

null_counts = monthly_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in monthly_df.columns
])
display(null_counts)

# Log-transform — Brent and USDINR (always positive)
monthly_df = (
    monthly_df
    .withColumn("log_Brent_USD_bbl", log(col("Brent_USD_bbl")))
    .withColumn("log_USDINR",        log(col("USDINR")))
    .withColumn("log_BANKNIFTY_Close", log(col("BANKNIFTY_Close")))
)

# Log-transform — Liquidity with shift for negative values
liq_stats = monthly_df.selectExpr(
    "min(India_System_Liquidity_INR_Trn) as min_liq",
    "max(India_System_Liquidity_INR_Trn) as max_liq",
    "sum(case when India_System_Liquidity_INR_Trn <= 0 then 1 else 0 end) as negative_count"
).collect()[0]

print(f"Liquidity min:            {liq_stats['min_liq']:.4f}")
print(f"Liquidity max:            {liq_stats['max_liq']:.4f}")
print(f"Negative/zero months:     {liq_stats['negative_count']}")

liquidity_shift = float(abs(liq_stats["min_liq"]) + 1) if liq_stats["min_liq"] <= 0 else 0.0
print(f"Liquidity shift applied:  {liquidity_shift}")

monthly_df = monthly_df.withColumn(
    "log_India_System_Liquidity_INR_Trn",
    log(col("India_System_Liquidity_INR_Trn") + lit(liquidity_shift))
)

# Derived rate features
monthly_df = (
    monthly_df
    .withColumn("India_Real_Rate",
                col("India_Policy_Rate_Pct") - col("India_CPI_YoY_Pct"))
    .withColumn("India_Term_Spread",
                col("India_10Y_GSec_Yield_Pct") - col("India_Policy_Rate_Pct"))
    .withColumn("India_Rate_Differential",
                col("India_10Y_GSec_Yield_Pct") - col("US10Y_Yield_Pct"))
)

# Compute 3M forward log return target
w_target = Window.partitionBy(lit(1)).orderBy("Date")
monthly_df = monthly_df.withColumn(
    target_col,
    log(lead("BANKNIFTY_Close", 3).over(w_target) / col("BANKNIFTY_Close"))
)

# Select final clean columns
final_cols = (
    ["Date", "BANKNIFTY_Close", "log_BANKNIFTY_Close", target_col]
    + macro_features
)
monthly_df = monthly_df.select(*final_cols)

# Null audit before dropna
total_rows = monthly_df.count()
print(f"Rows before dropna: {total_rows}")

null_counts = monthly_df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in monthly_df.columns
])
display(null_counts)

monthly_df = monthly_df.dropna()
final_rows = monthly_df.count()
print(f"Rows after dropna:  {final_rows}")
print(f"Rows dropped:       {total_rows - final_rows}")
print(f"  — expected ~3 from forward return window (lead=3)")

display(monthly_df.limit(10))


# COMMAND ----------

# Write silver table
monthly_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.silver_macro_base"
)

# Store liquidity shift as table property for downstream reference
spark.sql(f"""
    ALTER TABLE {catalog}.{schema}.silver_macro_base
    SET TBLPROPERTIES ('liquidity_shift' = '{liquidity_shift}')
""")

print(f"Silver table created:     {catalog}.{schema}.silver_macro_base")
print(f"Monthly observations:     {final_rows}")
print(f"Liquidity shift stored:   {liquidity_shift}")

# COMMAND ----------


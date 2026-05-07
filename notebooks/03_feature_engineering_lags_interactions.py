# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Feature Engineering — Lags, Forward Returns & Brent Interactions
# MAGIC Creates the 3M forward return target, lag features, direct Brent interactions, and lagged Brent transmission features.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, lead, lag, log, lit


# COMMAND ----------

# Load silver table

df = spark.table(f"{catalog}.{schema}.silver_macro_base")

w = Window.orderBy("Date")

# COMMAND ----------

# Forward return targets

df = df.withColumn(
    "BANKNIFTY_1M_Forward_Return",
    (lead("BANKNIFTY_Close", 1).over(w) - col("BANKNIFTY_Close")) / col("BANKNIFTY_Close")
)

df = df.withColumn(
    "BANKNIFTY_3M_Forward_Return",
    (lead("BANKNIFTY_Close", 3).over(w) - col("BANKNIFTY_Close")) / col("BANKNIFTY_Close")
)

df = df.withColumn(
    "BANKNIFTY_6M_Forward_Return",
    (lead("BANKNIFTY_Close", 6).over(w) - col("BANKNIFTY_Close")) / col("BANKNIFTY_Close")
)

# COMMAND ----------

# Log-return transformations

df = df.withColumn(
    "Brent_Log_Return_1M",
    log(col("Brent_USD_bbl") / lag("Brent_USD_bbl", 1).over(w))
)

df = df.withColumn(
    "Brent_Log_Return_3M",
    log(col("Brent_USD_bbl") / lag("Brent_USD_bbl", 3).over(w))
)

df = df.withColumn(
    "USDINR_Log_Return_1M",
    log(col("USDINR") / lag("USDINR", 1).over(w))
)

df = df.withColumn(
    "USDINR_Log_Return_3M",
    log(col("USDINR") / lag("USDINR", 3).over(w))
)

# COMMAND ----------

# Liquidity transformation

min_liquidity = df.selectExpr(
    "min(India_System_Liquidity_INR_Trn) as min_liquidity"
).collect()[0]["min_liquidity"]

liquidity_shift = abs(min_liquidity) + 1 if min_liquidity <= 0 else 0

df = df.withColumn(
    "Log_System_Liquidity",
    log(col("India_System_Liquidity_INR_Trn") + lit(liquidity_shift))
)

# COMMAND ----------

# Lag features

fast_lags = [1, 3, 6]
slow_lags = [3, 6, 12]

fast_vars = [
    "Brent_Log_Return_1M",
    "Brent_Log_Return_3M",
    "USDINR_Log_Return_1M",
    "USDINR_Log_Return_3M",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
    "Log_System_Liquidity"
]

slow_vars = [
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct"
]

for var in fast_vars:
    for l in fast_lags:
        df = df.withColumn(f"{var}_lag_{l}", lag(var, l).over(w))

for var in slow_vars:
    for l in slow_lags:
        df = df.withColumn(f"{var}_lag_{l}", lag(var, l).over(w))

# COMMAND ----------

# Brent interaction terms

interaction_vars = [
    "USDINR_Log_Return_1M",
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_10Y_GSec_Yield_Pct",
    "Log_System_Liquidity",
    "India_Bank_Credit_Growth_YoY_Pct"
]

for var in interaction_vars:
    df = df.withColumn(
        f"BrentReturn_x_{var}",
        col("Brent_Log_Return_1M") * col(var)
    )

# COMMAND ----------

# Lagged Brent interactions

df = df.withColumn(
    "BrentReturn_lag3_x_CPI_lag3",
    col("Brent_Log_Return_1M_lag_3") * col("India_CPI_YoY_Pct_lag_3")
)

df = df.withColumn(
    "BrentReturn_lag6_x_PolicyRate_lag6",
    col("Brent_Log_Return_1M_lag_6") * col("India_Policy_Rate_Pct_lag_6")
)

df = df.withColumn(
    "BrentReturn_lag3_x_USDINRReturn_lag1",
    col("Brent_Log_Return_1M_lag_3") * col("USDINR_Log_Return_1M_lag_1")
)

# COMMAND ----------

# Remove nulls

df = df.dropna()

# COMMAND ----------

# Save gold table

df.write.mode("overwrite").format("delta").saveAsTable(
    f"{catalog}.{schema}.gold_macro_features"
)

display(df)

# COMMAND ----------

print("Feature engineering completed.")
print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")

# COMMAND ----------


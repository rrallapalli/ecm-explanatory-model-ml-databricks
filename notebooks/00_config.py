# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Config
# MAGIC Shared configuration for the BANKNIFTY macro and Brent interaction modeling pipeline.

# COMMAND ----------

catalog = "workspace"
schema = "banknifty_macro"

input_file_path = "/Volumes/workspace/banknifty_macro/filestore/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes (1).csv"

target_col = "BANKNIFTY_3M_Forward_Return"

macro_features = [
    "Brent_USD_bbl",
    "USDINR",
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct",
    "India_System_Liquidity_INR_Trn"
]

exclude_cols = [
    "NIFTY_Close",
    "BANKNIFTY_to_NIFTY_Ratio",
    "India_Bank_PB_Multiple",
    "Shock_Regime",
    "Shock_Intensity"
]

fast_vars = [
    "Brent_USD_bbl",
    "USDINR",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
    "India_System_Liquidity_INR_Trn"
]

slow_vars = [
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct"
]

fast_lags = [1, 3, 6]
slow_lags = [3, 6, 12]

interaction_vars = [
    "USDINR",
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_10Y_GSec_Yield_Pct",
    "India_System_Liquidity_INR_Trn",
    "India_Bank_Credit_Growth_YoY_Pct"
]

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Using catalog.schema: {catalog}.{schema}")
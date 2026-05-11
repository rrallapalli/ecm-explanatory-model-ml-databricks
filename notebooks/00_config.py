# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Config
# MAGIC Shared configuration for the BANKNIFTY macro and Brent interaction modeling pipeline.

# COMMAND ----------

# MAGIC Central configuration for all pipeline notebooks.

catalog = "workspace"
schema  = "banknifty_macro"

input_file_path = "/Volumes/workspace/banknifty_macro/filestore/india_nifty_banknifty_synthetic_macro_sector_daily_with_shock_regimes (1).csv"

target_col = "BANKNIFTY_3M_Forward_Log_Return"

# Raw columns required from bronze
raw_required_cols = [
    "Date",
    "BANKNIFTY_Close",
    "Brent_USD_bbl",
    "USDINR",
    "India_System_Liquidity_INR_Trn",
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct",
]

# Resampling strategy
# fast_market_vars  — monthly mean (volatile, market-driven)
# slow_macro_vars   — last value of month (release-driven)
fast_market_vars = [
    "Brent_USD_bbl",
    "USDINR",
    "India_System_Liquidity_INR_Trn",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
]

slow_macro_vars = [
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct",
]

# Log-transform these after resampling (level/price series)
log_transform_cols = [
    "Brent_USD_bbl",
    "USDINR",
    "India_System_Liquidity_INR_Trn",  # shift applied in silver if negative
]

# Derived rate features (computed in silver, originals dropped)
# India_Real_Rate         = India_Policy_Rate_Pct - India_CPI_YoY_Pct
# India_Term_Spread       = India_10Y_GSec_Yield_Pct - India_Policy_Rate_Pct
# India_Rate_Differential = India_10Y_GSec_Yield_Pct - US10Y_Yield_Pct

# Final clean feature set for modeling
macro_features = [
    "log_Brent_USD_bbl",
    "log_USDINR",
    "log_India_System_Liquidity_INR_Trn",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct",
    "India_Real_Rate",
    "India_Term_Spread",
    "India_Rate_Differential",
]

# Lag structure — monthly periods
fast_vars = [
    "log_Brent_USD_bbl",
    "log_USDINR",
    "log_India_System_Liquidity_INR_Trn",
    "India_Rate_Differential",
    "India_Term_Spread",
]

slow_vars = [
    "India_Real_Rate",
    "India_Real_GDP_YoY_Pct",
    "India_Bank_Credit_Growth_YoY_Pct",
]

fast_lags = [1, 3, 6]   # months
slow_lags = [3, 6, 12]  # months

# Momentum periods for log-transformed level vars
momentum_vars = {
    "log_Brent_USD_bbl":                  "Brent",
    "log_USDINR":                         "USDINR",
    "log_India_System_Liquidity_INR_Trn": "Liquidity",
}
momentum_periods = [1, 3]  # months

# Interaction terms — derived features only, no raw originals
interaction_vars = [
    "log_USDINR",
    "log_India_System_Liquidity_INR_Trn",
    "India_Real_Rate",
    "India_Term_Spread",
    "India_Rate_Differential",
    "India_Bank_Credit_Growth_YoY_Pct",
]

# Columns to exclude from modeling
exclude_cols = [
    "NIFTY_Close",
    "BANKNIFTY_to_NIFTY_Ratio",
    "India_Bank_PB_Multiple",
    "Shock_Regime",
    "Shock_Intensity",
    # originals absorbed into derived features
    "India_CPI_YoY_Pct",
    "India_Policy_Rate_Pct",
    "India_10Y_GSec_Yield_Pct",
    "US10Y_Yield_Pct",
]

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Using catalog.schema: {catalog}.{schema}")
# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # 03 Feature Engineering — Lags, Forward Returns & Brent Interactions
# MAGIC
# MAGIC ## Overview
# MAGIC Creates lag features, momentum indicators, and interaction terms from the silver macro base table to produce the final gold feature set for BANKNIFTY 3-month forward return prediction.
# MAGIC
# MAGIC ## Input
# MAGIC * **Source**: `silver_macro_base` table
# MAGIC * **Target variable**: `Return_3M_Forward_Log` (already created in silver)
# MAGIC * **Frequency**: Monthly observations
# MAGIC * **Metadata**: Retrieves `liquidity_shift` from table properties for consistency
# MAGIC
# MAGIC ## Feature Engineering Pipeline
# MAGIC
# MAGIC ### 1. Momentum Features
# MAGIC * **Definition**: Log return over k periods = `log(P_t) - log(P_t-k)`
# MAGIC * **Variables**: BANKNIFTY, Brent, USDINR (from config `momentum_vars`)
# MAGIC * **Periods**: Multiple lookback windows (from config `momentum_periods`)
# MAGIC * **Purpose**: Captures short-to-medium term price trends and reversions
# MAGIC * **Implementation**: Simple subtraction since log columns already exist in silver
# MAGIC
# MAGIC ### 2. Fast Variable Lags
# MAGIC * **Variables**: Market-sensitive indicators (USDINR, Brent)
# MAGIC * **Lag structure**: Short lags (1-3 months from config `fast_lags`)
# MAGIC * **Rationale**: Fast-moving variables have immediate transmission effects
# MAGIC * **Window**: Ordered by Date, single partition
# MAGIC
# MAGIC ### 3. Slow Variable Lags
# MAGIC * **Variables**: Macro fundamentals (policy rates, yields, CPI, credit growth, liquidity)
# MAGIC * **Lag structure**: Longer lags (3-12 months from config `slow_lags`)
# MAGIC * **Rationale**: Policy and structural variables have delayed transmission to markets
# MAGIC * **Expected data loss**: ~12 rows from maximum lag window
# MAGIC
# MAGIC ### 4. Contemporaneous Interactions
# MAGIC * **Structure**: Brent 1-month momentum × each interaction variable
# MAGIC * **Variables**: From config `interaction_vars`
# MAGIC * **Purpose**: Captures immediate oil shock amplification through macro channels
# MAGIC * **Timing**: No lag - measures immediate correlation
# MAGIC
# MAGIC ### 5. Lagged Interaction Terms (Transmission Channels)
# MAGIC Five strategically designed interactions capturing delayed transmission:
# MAGIC
# MAGIC | Interaction | Lag Structure | Economic Channel |
# MAGIC | --- | --- | --- |
# MAGIC | BrentLag3 × RealRateLag3 | 3M + 3M | Oil shock → inflation → policy rate adjustment |
# MAGIC | BrentLag3 × LiquidityLag3 | 3M + 3M | Oil shock → liquidity conditions |
# MAGIC | USDINRLag1 × CreditLag3 | 1M + 3M | Currency depreciation → credit availability |
# MAGIC | TermSpreadLag3 × CreditLag3 | 3M + 3M | Yield curve shape → credit conditions |
# MAGIC | RealRateLag3 × CreditLag6 | 3M + 6M | Monetary policy → lending transmission |
# MAGIC
# MAGIC ### 6. Data Cleaning
# MAGIC * **Null audit**: Pre-dropna count by column displayed
# MAGIC * **Drop strategy**: Remove any row with null values
# MAGIC * **Expected loss**: ~12 rows from maximum slow lag window (12 months)
# MAGIC * **Final validation**: Row count and feature count summary
# MAGIC
# MAGIC ## Output
# MAGIC * **Table**: `gold_macro_features`
# MAGIC * **Write mode**: Overwrite
# MAGIC * **Format**: Delta
# MAGIC * **Feature count**: Base macro + momentum + lags + interactions (~80-100 features)
# MAGIC * **Target**: `Return_3M_Forward_Log` (3-month forward log return)
# MAGIC
# MAGIC ## Feature Summary Structure
# MAGIC * Base macro features (from silver)
# MAGIC * Momentum features (3 variables × multiple periods)
# MAGIC * Fast lag features (2 variables × short lags)
# MAGIC * Slow lag features (~10 variables × long lags)
# MAGIC * Contemporaneous interactions (Brent1M × N vars)
# MAGIC * Lagged interactions (5 strategically designed)
# MAGIC
# MAGIC ## Key Validations
# MAGIC * Schema validation against expected silver columns
# MAGIC * Liquidity shift consistency check
# MAGIC * Null counts pre/post cleaning
# MAGIC * Feature count reconciliation
# MAGIC * Sample preview of final feature set
# MAGIC
# MAGIC ## Notebook Organization
# MAGIC * **Cell 3**: Data Loading & Validation
# MAGIC * **Cell 4**: Momentum Features
# MAGIC * **Cell 5**: Lag Features (Fast + Slow)
# MAGIC * **Cell 6**: Interaction Terms (Contemporaneous + Lagged)
# MAGIC * **Cell 7**: Null Audit, Cleaning & Summary
# MAGIC * **Cell 8**: Write Gold Table

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Data Loading & Validation
from pyspark.sql.functions import col, lag, lit, count, when
from pyspark.sql.window import Window

print("="*60)
print("STEP 1: Data Loading & Validation")
print("="*60)

# Load silver table
df = spark.table(f"{catalog}.{schema}.silver_macro_base")
print(f"\nLoaded silver table: {df.count()} rows")

# Define window for lag operations
w = Window.partitionBy(lit(1)).orderBy("Date")

# Retrieve liquidity shift from silver table properties
props = (
    spark.sql(f"SHOW TBLPROPERTIES {catalog}.{schema}.silver_macro_base")
    .filter(col("key") == "liquidity_shift")
    .collect()
)
liquidity_shift = float(props[0]["value"]) if props else 0.0
print(f"Liquidity shift retrieved: {liquidity_shift}")

# Validate expected columns from silver
expected_from_silver = (
    ["Date", "BANKNIFTY_Close", "log_BANKNIFTY_Close", target_col]
    + macro_features
)
missing = [c for c in expected_from_silver if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns from silver: {missing}")

print(f"\n✓ Schema validated — {len(df.columns)} columns present")
print(f"  Base macro features:     {len(macro_features)}")
print(f"  Target column:           {target_col}")
print(f"  Date range:              {df.select('Date').agg({'Date': 'min'}).collect()[0][0]} to {df.select('Date').agg({'Date': 'max'}).collect()[0][0]}")

# COMMAND ----------

# DBTITLE 1,Momentum Features
print("="*60)
print("STEP 2: Momentum Features")
print("="*60)
print("Definition: log(P_t) - log(P_t-k) = log return over k periods\n")

# Momentum features
# log(P_t) - log(P_t-k) = log return over k periods
# silver already has log columns so this is a simple subtraction
momentum_count = 0
for col_name, label in momentum_vars.items():
    for k in momentum_periods:
        df = df.withColumn(
            f"{label}_Momentum_{k}M",
            col(col_name) - lag(col_name, k).over(w)
        )
        momentum_count += 1
        print(f"  Created: {label}_Momentum_{k}M")

print(f"\n✓ Momentum features created: {momentum_count}")
print(f"  Variables:  {list(momentum_vars.values())}")
print(f"  Periods:    {momentum_periods}")

# COMMAND ----------

# DBTITLE 1,Lag Features
print("="*60)
print("STEP 3: Lag Features")
print("="*60)

# Fast variable lags
print("\nFast variable lags (market-sensitive indicators):")
fast_count = 0
for var in fast_vars:
    for l in fast_lags:
        df = df.withColumn(f"{var}_lag_{l}", lag(var, l).over(w))
        fast_count += 1
        print(f"  Created: {var}_lag_{l}")

print(f"\n✓ Fast lag features created: {fast_count}")
print(f"  Variables:  {fast_vars}")
print(f"  Lags:       {fast_lags}")

# Slow variable lags
print("\n" + "="*60)
print("Slow variable lags (macro fundamentals):")
print("="*60)
slow_count = 0
for var in slow_vars:
    for l in slow_lags:
        df = df.withColumn(f"{var}_lag_{l}", lag(var, l).over(w))
        slow_count += 1

print(f"\n✓ Slow lag features created: {slow_count}")
print(f"  Variables:  {len(slow_vars)} macro fundamentals")
print(f"  Lags:       {slow_lags}")
print(f"  Max lag:    {max(slow_lags)} months (will lose ~{max(slow_lags)} rows)")

# COMMAND ----------

# DBTITLE 1,Interaction Terms
print("="*60)
print("STEP 4: Interaction Terms")
print("="*60)

# Contemporaneous interaction terms
print("\nContemporneous interactions (Brent 1M momentum × variables):")
contemp_count = 0
for var in interaction_vars:
    df = df.withColumn(
        f"Brent1M_x_{var}",
        col("Brent_Momentum_1M") * col(var)
    )
    contemp_count += 1
    print(f"  Created: Brent1M_x_{var}")

print(f"\n✓ Contemporaneous interactions created: {contemp_count}")

# Lagged interaction terms
print("\n" + "="*60)
print("Lagged interaction terms (transmission channels):")
print("="*60)

lagged_interactions = [
    # Brent shock feeds into real rate with a lag
    ("log_Brent_USD_bbl_lag_3",  "India_Real_Rate_lag_3",
     "BrentLag3_x_RealRateLag3",
     "Oil shock → inflation → policy rate"),

    # Brent shock + liquidity conditions
    ("log_Brent_USD_bbl_lag_3",  "log_India_System_Liquidity_INR_Trn_lag_3",
     "BrentLag3_x_LiquidityLag3",
     "Oil shock → liquidity conditions"),

    # Currency + credit growth transmission
    ("log_USDINR_lag_1",         "India_Bank_Credit_Growth_YoY_Pct_lag_3",
     "USDINRLag1_x_CreditLag3",
     "Currency depreciation → credit"),

    # Term spread + credit
    ("India_Term_Spread_lag_3",  "India_Bank_Credit_Growth_YoY_Pct_lag_3",
     "TermSpreadLag3_x_CreditLag3",
     "Yield curve → credit conditions"),

    # Real rate + credit
    ("India_Real_Rate_lag_3",    "India_Bank_Credit_Growth_YoY_Pct_lag_6",
     "RealRateLag3_x_CreditLag6",
     "Monetary policy → lending"),
]

for col_a, col_b, new_name, channel in lagged_interactions:
    df = df.withColumn(new_name, col(col_a) * col(col_b))
    print(f"  Created: {new_name}")
    print(f"    Channel: {channel}")

print(f"\n✓ Lagged interactions created: {len(lagged_interactions)}")

# COMMAND ----------

# DBTITLE 1,Null Audit, Cleaning & Summary
print("="*60)
print("STEP 5: Null Audit, Cleaning & Summary")
print("="*60)

# Null audit before dropna
total_rows = df.count()
print(f"\nRows before dropna: {total_rows}")

print("\nNull counts by column:")
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])
display(null_counts)

# Drop rows with any nulls
df = df.dropna()
final_rows = df.count()
rows_dropped = total_rows - final_rows

print(f"\nRows after dropna:  {final_rows}")
print(f"Rows dropped:       {rows_dropped}")
print(f"Expected loss:      ~{max(slow_lags)} from max slow lag window")

if abs(rows_dropped - max(slow_lags)) > 5:
    print(f"  ⚠ WARNING: Row loss ({rows_dropped}) differs from expected (~{max(slow_lags)})")
else:
    print(f"  ✓ Row loss consistent with lag window")

# Feature summary
print(f"\n{'='*60}")
print(f"Feature Engineering Summary")
print(f"{'='*60}")

total_features = len(df.columns) - 4  # exclude Date, BANKNIFTY_Close, log_BANKNIFTY_Close, target
print(f"Total modeling features:    {total_features}")
print(f"  Base macro features:      {len(macro_features)}")
print(f"  Momentum features:        {len(momentum_vars) * len(momentum_periods)}")
print(f"  Fast lag features:        {len(fast_vars) * len(fast_lags)}")
print(f"  Slow lag features:        {len(slow_vars) * len(slow_lags)}")
print(f"  Contemporaneous ix:       {len(interaction_vars)}")
print(f"  Lagged interactions:      {len(lagged_interactions)}")
print(f"\nFinal dataset:")
print(f"  Rows:                     {final_rows}")
print(f"  Columns:                  {len(df.columns)}")
print(f"  Target:                   {target_col}")

print(f"\nSample preview (first 10 rows):")
display(df.limit(10))

print(f"\n{'='*60}")
print(f"Feature engineering complete")
print(f"{'='*60}")

# COMMAND ----------

# DBTITLE 1,Write Gold Table
print("="*60)
print("STEP 6: Write Gold Table")
print("="*60)

# Write gold table
df.write.mode("overwrite").format("delta").saveAsTable(
    f"{catalog}.{schema}.gold_macro_features"
)

print(f"\n✓ Gold table created: {catalog}.{schema}.gold_macro_features")
print(f"  Rows:                {df.count()}")
print(f"  Columns:             {len(df.columns)}")
print(f"  Target:              {target_col}")
print(f"  Total features:      {len(df.columns) - 4}")
print(f"\nReady for:")
print(f"  → 04_correlation_feature_selection")
print(f"  → 05_model_training")

# COMMAND ----------


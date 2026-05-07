# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Correlation & Feature Selection
# MAGIC Uses correlation for EDA, lag signal discovery, and initial feature screening. This is not the only feature selection method.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np

model_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
model_df = model_df.sort_values("Date")

drop_cols = [
    "Date",
    "BANKNIFTY_Close",
    "BANKNIFTY_1M_Forward_Return",
    "BANKNIFTY_3M_Forward_Return",
    "BANKNIFTY_6M_Forward_Return",
    target_col
]

feature_cols = [c for c in model_df.columns if c not in drop_cols]

corr_df = (
    model_df[feature_cols + [target_col]]
    .corr(numeric_only=True)[[target_col]]
    .reset_index()
    .rename(columns={"index": "feature", target_col: "correlation_with_target"})
)

corr_df["abs_correlation"] = corr_df["correlation_with_target"].abs()
corr_df = corr_df.sort_values("abs_correlation", ascending=False)

display(corr_df)

# COMMAND ----------

spark.createDataFrame(corr_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_correlation_summary"
)

# COMMAND ----------

# Pairwise correlation matrix for multicollinearity inspection
pairwise_corr = model_df[feature_cols].corr(numeric_only=True)

high_corr_pairs = []
for i, col1 in enumerate(pairwise_corr.columns):
    for j, col2 in enumerate(pairwise_corr.columns):
        if i < j:
            corr_value = pairwise_corr.loc[col1, col2]
            if abs(corr_value) >= 0.80:
                high_corr_pairs.append((col1, col2, corr_value, abs(corr_value)))

high_corr_df = pd.DataFrame(high_corr_pairs, columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
high_corr_df = high_corr_df.sort_values("abs_correlation", ascending=False)

display(high_corr_df)


# COMMAND ----------

spark.createDataFrame(high_corr_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_high_correlation_pairs"
)

# COMMAND ----------


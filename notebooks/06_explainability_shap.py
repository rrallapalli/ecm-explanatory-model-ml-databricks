# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Explainability — SHAP
# MAGIC Uses SHAP to explain macro, lag, and Brent interaction effects.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install shap numpy --upgrade

# COMMAND ----------

import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

model_df = spark.table(f"{catalog}.{schema}.gold_macro_features").toPandas()
model_df = model_df.sort_values("Date")

drop_cols = [
    "Date",
    "BANKNIFTY_Close",
    "BANKNIFTY_1M_Forward_Return",
    "BANKNIFTY_6M_Forward_Return",
    target_col
]

feature_cols = [c for c in model_df.columns if c not in drop_cols]

X = model_df[feature_cols]
y = model_df[target_col]

split_idx = int(len(model_df) * 0.80)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# COMMAND ----------

shap.summary_plot(shap_values, X_test)

# COMMAND ----------

import pandas as pd

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

display(shap_importance_df)

spark.createDataFrame(shap_importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_shap_feature_importance"
)

# COMMAND ----------

brent_shap_df = shap_importance_df[shap_importance_df["feature"].str.contains("Brent", case=False, regex=False)]

display(brent_shap_df)

spark.createDataFrame(brent_shap_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_brent_shap_importance"
)

# COMMAND ----------


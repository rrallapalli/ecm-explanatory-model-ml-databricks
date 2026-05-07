# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Model Diagnostics
# MAGIC Creates diagnostic tables for model stability, error behavior, residual bias, and return regime performance.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions_df = spark.table(f"{catalog}.{schema}.gold_model_predictions").toPandas()
predictions_df = predictions_df.sort_values("Date")

predictions_df["Residual"] = (
    predictions_df["Actual_3M_Forward_Return"] - predictions_df["Predicted_3M_Forward_Return"]
)

predictions_df["Absolute_Error"] = predictions_df["Residual"].abs()
predictions_df["Squared_Error"] = predictions_df["Residual"] ** 2

predictions_df["Actual_Direction"] = np.where(predictions_df["Actual_3M_Forward_Return"] > 0, 1, 0)
predictions_df["Predicted_Direction"] = np.where(predictions_df["Predicted_3M_Forward_Return"] > 0, 1, 0)
predictions_df["Direction_Correct"] = np.where(
    predictions_df["Actual_Direction"] == predictions_df["Predicted_Direction"], 1, 0
)

predictions_df["Residual_Bias"] = np.where(
    predictions_df["Residual"] > 0,
    "Under_Prediction",
    "Over_Prediction"
)

predictions_df["Actual_Return_Regime"] = pd.cut(
    predictions_df["Actual_3M_Forward_Return"],
    bins=[-np.inf, -0.05, 0.05, np.inf],
    labels=["Negative", "Flat", "Positive"]
).astype(str)

display(predictions_df)

spark.createDataFrame(predictions_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_diagnostics"
)

# COMMAND ----------

mae = mean_absolute_error(predictions_df["Actual_3M_Forward_Return"], predictions_df["Predicted_3M_Forward_Return"])
rmse = mean_squared_error(predictions_df["Actual_3M_Forward_Return"], predictions_df["Predicted_3M_Forward_Return"], squared=False)
r2 = r2_score(predictions_df["Actual_3M_Forward_Return"], predictions_df["Predicted_3M_Forward_Return"])
directional_accuracy = predictions_df["Direction_Correct"].mean()

metrics_df = pd.DataFrame({
    "model_name": [predictions_df["model_name"].iloc[0]],
    "target": [target_col],
    "mae": [mae],
    "rmse": [rmse],
    "r2": [r2],
    "directional_accuracy": [directional_accuracy],
    "mean_residual": [predictions_df["Residual"].mean()],
    "residual_std": [predictions_df["Residual"].std()],
    "mean_absolute_error_pct": [predictions_df["Absolute_Error"].mean()]
})

display(metrics_df)

spark.createDataFrame(metrics_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_metrics"
)

# COMMAND ----------

bias_summary = (
    predictions_df
    .groupby("Residual_Bias")
    .agg(
        count=("Residual", "count"),
        avg_residual=("Residual", "mean"),
        avg_abs_error=("Absolute_Error", "mean")
    )
    .reset_index()
)

display(bias_summary)

spark.createDataFrame(bias_summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_residual_bias_summary"
)

# COMMAND ----------

regime_error = (
    predictions_df
    .groupby("Actual_Return_Regime")
    .agg(
        count=("Actual_3M_Forward_Return", "count"),
        avg_actual_return=("Actual_3M_Forward_Return", "mean"),
        avg_predicted_return=("Predicted_3M_Forward_Return", "mean"),
        avg_abs_error=("Absolute_Error", "mean"),
        directional_accuracy=("Direction_Correct", "mean")
    )
    .reset_index()
)

display(regime_error)

spark.createDataFrame(regime_error).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_regime_diagnostics"
)

# COMMAND ----------


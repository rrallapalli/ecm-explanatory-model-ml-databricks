# Databricks notebook source
# DBTITLE 1,Model Training Header
# MAGIC %md
# MAGIC # 05 Model Training & Ensemble
# MAGIC Trains multiple regression models (Lasso, Ridge, Random Forest, XGBoost, MLP Neural Network) to predict 3-month forward Bank Nifty returns using macro features, lags, and interactions.
# MAGIC
# MAGIC Each model is wrapped in a **sklearn Pipeline** with **StandardScaler** preprocessing, ensuring:
# MAGIC * Features are scaled to zero mean and unit variance
# MAGIC * Scaler is fitted only on training data (no data leakage)
# MAGIC * Models are production-ready with preprocessing built-in

# COMMAND ----------

# DBTITLE 1,Load Config
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Import Libraries
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost not installed, will skip XGBoost model")
    XGBRegressor = None

# COMMAND ----------

# DBTITLE 1,Load and Prepare Data
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
test_dates = model_df.iloc[split_idx:]["Date"]

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Feature count: {len(feature_cols)}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Shared/banknifty_macro_brent_interaction"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Model Evaluation Function
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    direction_acc = (np.sign(y_test.values) == np.sign(preds)).mean()
    return model, preds, mae, rmse, r2, direction_acc

# COMMAND ----------

# DBTITLE 1,Train Models
model_results = []

models = {
    "ridge_macro_lags_interactions": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "lasso_feature_selection": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ]),
    "random_forest_macro_interaction_model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ))
    ]),
    "mlp_neural_network": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        ))
    ])
}

# Add XGBoost if available
if XGBRegressor is not None:
    models["xgboost_macro_interaction_model"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ))
    ])

best_model = None
best_preds = None
best_rmse = float("inf")
best_model_name = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        fitted_model, preds, mae, rmse, r2, direction_acc = evaluate_model(
            model_name, model, X_train, X_test, y_train, y_test
        )

        mlflow.log_param("target", target_col)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("directional_accuracy", direction_acc)

        # Infer signature for Unity Catalog registration
        signature = mlflow.models.infer_signature(X_train, preds)

        mlflow.sklearn.log_model(
            fitted_model,
            artifact_path=model_name,
            registered_model_name=f"{catalog}.{schema}.{model_name}",
            signature=signature
        )

        model_results.append({
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": direction_acc
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = fitted_model
            best_preds = preds
            best_model_name = model_name

results_df = pd.DataFrame(model_results).sort_values("rmse")
display(results_df)

spark.createDataFrame(results_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_comparison"
)

# COMMAND ----------

# DBTITLE 1,Save Predictions and Feature Importance
# Persist best model diagnostics base output for later notebooks
predictions_df = pd.DataFrame({
    "Date": test_dates.values,
    "Actual_3M_Forward_Return": y_test.values,
    "Predicted_3M_Forward_Return": best_preds,
    "model_name": best_model_name
})

spark.createDataFrame(predictions_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_model_predictions"
)

# Feature importance for Random Forest if it is available
rf_model = models["random_forest_macro_interaction_model"]
rf_model.fit(X_train, y_train)

# Access the Random Forest model from the Pipeline
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.named_steps['model'].feature_importances_
}).sort_values("importance", ascending=False)

display(importance_df)

spark.createDataFrame(importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_feature_importance"
)

brent_importance_df = importance_df[importance_df["feature"].str.contains("Brent", case=False, regex=False)]

spark.createDataFrame(brent_importance_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog}.{schema}.gold_brent_feature_importance"
)

# COMMAND ----------


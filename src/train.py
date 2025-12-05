"""
Train model, log params/metrics/artifacts to MLflow, register model locally.
"""
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from src.utils import PROCESSED_DIR, set_mlflow_tracking, save_model_local
from src.features import fit_transform_features
import joblib
import os

def train_and_log(n_estimators=100, max_depth=None, random_state=42):
    set_mlflow_tracking()
    mlflow.set_experiment("california_housing_experiment")
    with mlflow.start_run() as run:
        # load data
        train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
        test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        target = "MedHouseVal" if "MedHouseVal" in train.columns else train.columns[-1]

        # choose numeric features (all except target)
        numeric_features = [c for c in train.columns if c != target]

        # feature engineering
        X_train, pipeline = fit_transform_features(train, numeric_features)
        y_train = train[target].values
        # transform test using saved pipeline
        X_test = pipeline.transform(test[numeric_features])
        y_test = test[target].values

        # model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)

        # predict and metrics
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        # log to mlflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
 
        # save pipeline and model as artifacts
        model_artifact_path = PROCESSED_DIR / "model.joblib"
        pipeline_artifact_path = PROCESSED_DIR / "feature_pipeline.joblib"

        joblib.dump(model, model_artifact_path)
        joblib.dump(pipeline, pipeline_artifact_path)

        mlflow.log_artifact(str(model_artifact_path), artifact_path="artifacts")
        mlflow.log_artifact(str(pipeline_artifact_path), artifact_path="artifacts")


        # register model as an MLflow model (local)
        # For local usage we can just register as model artifact path and later load from mlruns.
        print(f"Run id: {run.info.run_id}")
        print(f"Metrics: rmse={rmse}, r2={r2}")
        return {"run_id": run.info.run_id, "rmse": rmse, "r2": r2}
    
        metrics_path = PROCESSED_DIR / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump({"rmse": rmse, "r2": r2}, f, indent=4)

if __name__ == "__main__":
    train_and_log()

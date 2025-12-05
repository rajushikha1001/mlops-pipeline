"""
Simple FastAPI service that loads model artifacts from MLflow local runs and serves predictions.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import joblib
import numpy as np
import pandas as pd
from typing import List
from src.utils import set_mlflow_tracking, MLFLOW_TRACKING_URI, PROCESSED_DIR

app = FastAPI(title="mlops-pipeline-prediction")

set_mlflow_tracking()

class PredictRequest(BaseModel):
    data: List[dict]  # list of json rows

class PredictResponse(BaseModel):
    predictions: List[float]

def load_latest_artifacts():
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name("california_housing_experiment")
    if exp is None:
        raise RuntimeError("No experiment found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    run = runs[0]
    run_id = run.info.run_id

    # Correct file names (MLflow automatically goes inside "artifacts/")
    model_path = client.download_artifacts(run_id, "model.joblib")
    pipeline_path = client.download_artifacts(run_id, "feature_pipeline.joblib")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    return model, pipeline


_model = None
_pipeline = None

@app.on_event("startup")
def load_model():
    global _model, _pipeline
    _model, _pipeline = load_latest_artifacts()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global _model, _pipeline
    df = pd.DataFrame(req.data)
    # assume df has same columns as training numeric features
    X = df[_pipeline.transformers_[0][2]]
    X_trans = _pipeline.transform(X)
    preds = _model.predict(X_trans)
    return PredictResponse(predictions=preds.tolist())

@app.get("/")
def root():
    return {"status": "running"}

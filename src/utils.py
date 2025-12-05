import os
from pathlib import Path
import joblib
import mlflow

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_model_local(model, path: str):
    joblib.dump(model, path)

def load_model_local(path: str):
    return joblib.load(path)

def set_mlflow_tracking():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

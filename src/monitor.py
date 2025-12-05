"""
Monitoring job using Evidently. It loads test data, current model, and computes a drift & classification report.
Saves html report to PROCESSED_DIR/monitor_report.html
"""
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
import pandas as pd
from src.utils import PROCESSED_DIR, set_mlflow_tracking
import joblib
import mlflow

def run_monitoring():
    set_mlflow_tracking()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("california_housing_experiment")
    if exp is None:
        raise RuntimeError("No experiment found. Run training first.")
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    run = runs[0]
    run_id = run.info.run_id
    # download artifacts
    model_uri = client.download_artifacts(run_id, "artifacts/model.joblib")
    pipeline_uri = client.download_artifacts(run_id, "artifacts/feature_pipeline.joblib")
    model = joblib.load(model_uri)
    pipeline = joblib.load(pipeline_uri)

    # load test data
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    target = "MedHouseVal" if "MedHouseVal" in test.columns else test.columns[-1]
    numeric_features = [c for c in test.columns if c != target]
    X_test = test[numeric_features]
    y_test = test[target]

    # create dataset for Evidently: reference = train, current = test (for demo)
    # For simplicity we use test as current and also as reference — in production use real references
    reference = X_test.copy()
    reference[target] = y_test
    current = X_test.copy()
    current[target] = y_test

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    mapping = ColumnMapping(numerical_features=numeric_features, target=target)
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    html = report.as_html()
    out_path = PROCESSED_DIR / "monitor_report.html"
    out_path.write_text(html)
    print(f"Saved monitoring report to {out_path}")

if __name__ == "__main__":
    run_monitoring()

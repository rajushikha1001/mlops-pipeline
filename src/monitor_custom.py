import os
os.environ["PANDAS_IGNORE_PYARROW_IMPORT"] = "1"

import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import PROCESSED_DIR


# ----------------------------------------------------------
# CLEAN JSON SERIALIZATION FOR NUMPY / PANDAS
# ----------------------------------------------------------
def clean_json(obj):
    """Recursively convert numpy, pandas, sklearn, and other non-JSON types."""
    # Dictionaries
    if isinstance(obj, dict):
        return {clean_json(k): clean_json(v) for k, v in obj.items()}

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]

    # NumPy booleans
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # NumPy integers
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # NumPy floats
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas NA / NaT → convert to None
    if pd.isna(obj):
        return None

    return obj


# ----------------------------------------------------------
# POPULATION STABILITY INDEX
# ----------------------------------------------------------
def psi(expected, actual, buckets=10):
    """Population Stability Index calculation."""
    def scale_range(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-9)

    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_scaled = scale_range(expected, expected.min(), expected.max())
    actual_scaled = scale_range(actual, actual.min(), actual.max())

    expected_counts = np.histogram(expected_scaled, breakpoints)[0]
    actual_counts = np.histogram(actual_scaled, breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi_values = []
    for e, a in zip(expected_perc, actual_perc):
        if e == 0: e = 0.0001
        if a == 0: a = 0.0001
        psi_values.append((e - a) * np.log(e / a))

    return float(np.sum(psi_values))


# ----------------------------------------------------------
# MONITORING FUNCTION
# ----------------------------------------------------------
def monitor():
    # Load CSV data
    df_train = pd.read_csv(PROCESSED_DIR / "train.csv")
    df_test = pd.read_csv(PROCESSED_DIR / "test.csv")

    train_y = df_train["MedHouseVal"]
    test_y = df_test["MedHouseVal"]

    numeric_features = df_train.drop(columns=["MedHouseVal"]).columns

    drift_results = {}
    psi_results = {}
    data_quality = {}

    # ------------------------------------------------------
    # Compute Drift (KS test + PSI)
    # ------------------------------------------------------
    for col in numeric_features:
        ks_stat, ks_pvalue = ks_2samp(df_train[col], df_test[col])
        drift_results[col] = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "drift_detected": bool(ks_pvalue < 0.05)
        }
        psi_results[col] = psi(df_train[col].values, df_test[col].values)

    # ------------------------------------------------------
    # Compute data quality metrics
    # ------------------------------------------------------
    for col in numeric_features:
        data_quality[col] = {
            "missing_train": int(df_train[col].isna().sum()),
            "missing_test": int(df_test[col].isna().sum()),
            "mean_train": float(df_train[col].mean()),
            "mean_test": float(df_test[col].mean()),
            "std_train": float(df_train[col].std()),
            "std_test": float(df_test[col].std()),
        }

    # ------------------------------------------------------
    # Load prediction metrics (from training)
    # ------------------------------------------------------
    metrics_file = PROCESSED_DIR / "metrics_summary.json"

    if not metrics_file.exists():
        pred_metrics = {
            "rmse": None,
            "r2": None,
            "note": "Run training again to collect metrics."
        }
    else:
        pred_metrics = json.load(open(metrics_file))

    # ------------------------------------------------------
    # Build final monitoring report
    # ------------------------------------------------------
    report = {
        "data_drift_ks": drift_results,
        "data_drift_psi": psi_results,
        "data_quality": data_quality,
        "prediction_metrics": pred_metrics
    }

    # Output file paths
    json_path = PROCESSED_DIR / "monitor_report.json"
    html_path = PROCESSED_DIR / "monitor_report.html"

    # ------------------------------------------------------
    # Save JSON (clean)
    # ------------------------------------------------------
    with open(json_path, "w") as f:
        json.dump(clean_json(report), f, indent=4)

    # ------------------------------------------------------
    # Save HTML version
    # ------------------------------------------------------
    html_content = f"""
<html>
<body>
<h2>Monitoring Report</h2>
<pre>{json.dumps(clean_json(report), indent=4)}</pre>
</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Monitoring report saved:\n{json_path}\n{html_path}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    monitor()

import pandas as pd
from src.data_prep import prepare_and_save, load_raw_data
from src.features import fit_transform_features
from src.utils import PROCESSED_DIR

def test_feature_pipeline():
    prepare_and_save()
    df = load_raw_data()
    target = "MedHouseVal" if "MedHouseVal" in df.columns else df.columns[-1]
    numeric_features = [c for c in df.columns if c != target]
    X_trans, pipeline = fit_transform_features(df.head(50), numeric_features)
    assert X_trans.shape[0] == 50

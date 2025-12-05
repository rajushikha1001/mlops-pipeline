"""
Feature engineering: simple pipeline using sklearn: imputer, scaler, polynomial features (optional).
We expose fit_transform for training and transform for inference.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
from src.utils import PROCESSED_DIR

NUMERIC_PIPE = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    # ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # uncomment if desired
])

def build_feature_pipeline(numeric_features):
    preprocessor = ColumnTransformer([
        ("num", NUMERIC_PIPE, numeric_features)
    ])
    return preprocessor

def fit_transform_features(df: pd.DataFrame, numeric_features):
    pipeline = build_feature_pipeline(numeric_features)
    X = df[numeric_features]
    pipeline.fit(X)
    transformed = pipeline.transform(X)
    # save pipeline for inference
    joblib.dump(pipeline, PROCESSED_DIR / "feature_pipeline.joblib")
    return transformed, pipeline

def transform_features(df: pd.DataFrame):
    pipeline = joblib.load(PROCESSED_DIR / "feature_pipeline.joblib")
    X = df[pipeline.transformers_[0][2]]  # columns used
    return pipeline.transform(X)

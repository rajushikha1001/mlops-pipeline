"""
Load dataset, perform simple cleaning, split, and save to data/processed.
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils import ensure_dirs, PROCESSED_DIR
import os

def load_raw_data() -> pd.DataFrame:
    ds = fetch_california_housing(as_frame=True)
    df = pd.concat([ds.frame], axis=1)
    return df

def prepare_and_save(test_size=0.2, random_state=42):
    ensure_dirs()
    df = load_raw_data()
    # small cleaning example: drop rows with missing values (none expected here)
    df = df.dropna()
    # simple train-test split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    # save
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Saved train ({len(train)}) and test ({len(test)}) to {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_and_save()

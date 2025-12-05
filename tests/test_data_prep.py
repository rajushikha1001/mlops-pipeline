from src.data_prep import prepare_and_save
from src.utils import PROCESSED_DIR
import os

def test_prepare_and_save():
    prepare_and_save()
    assert (PROCESSED_DIR / "train.parquet").exists()
    assert (PROCESSED_DIR / "test.parquet").exists()

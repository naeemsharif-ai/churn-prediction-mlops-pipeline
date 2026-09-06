"""End-to-end data pipeline: raw CSV -> cleaned, feature-engineered, split parquet files."""
import logging

import pandas as pd

from churn.config import settings
from churn.data.clean import clean
from churn.data.load import download_raw_data
from churn.data.split import save_splits, stratified_split
from churn.data.validate import validate_raw
from churn.features.engineer import engineer_features

logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    raw_path = download_raw_data()
    df = pd.read_csv(raw_path)
    df = validate_raw(df)
    df = clean(df)
    df = engineer_features(df)
    train, val, test = stratified_split(df)
    save_splits(train, val, test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
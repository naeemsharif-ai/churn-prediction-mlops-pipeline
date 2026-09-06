"""Stratified train/val/test split, persisted as separate parquet files."""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import settings

logger = logging.getLogger(__name__)

def stratified_split(
    df: pd.DataFrame,
    target_col: str = settings.target_col,
    val_size: float = 0.15,
    test_size: float = 0.05,
    seed: int = settings.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val[target_col], random_state=seed
    )
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir = settings.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    logger.info(
        "Saved splits — train: %d, val: %d, test: %d", len(train), len(val), len(test)
    )
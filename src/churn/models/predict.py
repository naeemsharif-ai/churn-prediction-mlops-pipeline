"""Inference helpers — reused by the API in Stage 8."""
from pathlib import Path

import lightgbm as lgb
import pandas as pd


def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def predict_proba(model: lgb.Booster, X: pd.DataFrame) -> pd.Series:
    """Returns churn probability, aligned to X's index."""
    preds = model.predict(X, num_iteration=model.best_iteration)
    return pd.Series(preds, index=X.index, name="churn_probability")
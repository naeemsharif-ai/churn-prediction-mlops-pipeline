"""Model training: CV-based hyperparameter search + final fit with early stopping."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from churn.config import settings

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod", "tenure_bucket",
]


@dataclass
class TrainResult:
    model: lgb.Booster
    best_params: dict
    cv_auc_pr_mean: float
    cv_auc_pr_std: float
    val_auc_pr: float
    feature_names: list[str] = field(default_factory=list)


def _make_dataset(df: pd.DataFrame, target_col: str = settings.target_col) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _cv_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "seed": seed,
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CATEGORICAL_FEATURES)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set)

        booster = lgb.train(
            params, train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        fold_scores.append(average_precision_score(y_val, preds))

    return float(np.mean(fold_scores))


def tune_hyperparameters(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 30, seed: int = settings.random_seed
) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: _cv_objective(trial, X, y, seed), n_trials=n_trials, show_progress_bar=True)
    return study


def train_final_model(
    train_df: pd.DataFrame, val_df: pd.DataFrame, best_params: dict, seed: int = settings.random_seed
) -> TrainResult:
    X_train, y_train = _make_dataset(train_df)
    X_val, y_val = _make_dataset(val_df)

    params = {**best_params, "objective": "binary", "metric": "average_precision", "verbosity": -1, "seed": seed}

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set)

    model = lgb.train(
        params, train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)],
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc_pr = average_precision_score(y_val, val_preds)

    return TrainResult(
        model=model,
        best_params=best_params,
        cv_auc_pr_mean=0.0,  # filled by caller from study
        cv_auc_pr_std=0.0,
        val_auc_pr=val_auc_pr,
        feature_names=list(X_train.columns),
    )


def save_model(model: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)
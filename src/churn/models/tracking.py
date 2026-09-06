"""MLflow experiment tracking wrapper. Keeps train.py free of tracking-specific code."""
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd

from churn.config import settings

logger = logging.getLogger(__name__)


def init_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_optuna_study(study: optuna.Study) -> None:
    """Log every trial as a nested run, so the full search is visible in the UI,
    not just the winner — useful for showing tuning wasn't cherry-picked."""
    for trial in study.trials:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_auc_pr", trial.value)
            mlflow.set_tag("optuna_trial_state", str(trial.state))


def log_final_model(
    model: lgb.Booster,
    best_params: dict,
    cv_auc_pr_mean: float,
    val_metrics: dict[str, float],
    feature_names: list[str],
    model_path: Path,
) -> str:
    """Logs the final selected model as the parent run. Returns the run_id."""
    with mlflow.start_run(run_name="final_model") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_auc_pr_mean", cv_auc_pr_mean)
        for name, value in val_metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.set_tag("feature_names", ",".join(feature_names))

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn_lgbm",
        )
        mlflow.log_artifact(str(model_path))

        return run.info.run_id
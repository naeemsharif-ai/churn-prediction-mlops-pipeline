"""One-off training run entry point."""
import logging

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from churn.config import settings
from churn.models.train import tune_hyperparameters, train_final_model, save_model, _make_dataset
from churn.models.tracking import init_mlflow, log_optuna_study, log_final_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_val_metrics(model, val_df) -> dict[str, float]:
    X_val, y_val = _make_dataset(val_df)
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    pred_labels = (preds >= 0.5).astype(int)
    return {
        "val_auc_pr": average_precision_score(y_val, preds),
        "val_auc_roc": roc_auc_score(y_val, preds),
        "val_f1": f1_score(y_val, pred_labels),
    }


def main():
    init_mlflow()

    train_df = pd.read_parquet(settings.data_processed_dir / "train.parquet")
    val_df = pd.read_parquet(settings.data_processed_dir / "val.parquet")
    X_train, y_train = _make_dataset(train_df)

    with mlflow.start_run(run_name="hpo_search"):
        logger.info("Starting hyperparameter search (30 trials, 5-fold CV)...")
        study = tune_hyperparameters(X_train, y_train, n_trials=30)
        log_optuna_study(study)
        mlflow.log_metric("best_cv_auc_pr", study.best_value)

    logger.info("Best CV AUC-PR: %.4f", study.best_value)

    result = train_final_model(train_df, val_df, study.best_params)
    val_metrics = compute_val_metrics(result.model, val_df)
    logger.info("Val metrics: %s", val_metrics)

    model_path = settings.project_root / "models" / "lgbm_churn.txt"
    save_model(result.model, model_path)

    run_id = log_final_model(
        model=result.model,
        best_params=study.best_params,
        cv_auc_pr_mean=study.best_value,
        val_metrics=val_metrics,
        feature_names=result.feature_names,
        model_path=model_path,
    )
    logger.info("Final model run logged: %s", run_id)


if __name__ == "__main__":
    main()
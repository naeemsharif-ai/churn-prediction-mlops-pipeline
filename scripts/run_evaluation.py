"""Final, one-time test set evaluation. Run this once per candidate model, not iteratively."""
import logging

import mlflow
import pandas as pd
import json

from churn.config import settings
from churn.models.predict import load_model
from churn.models.train import _make_dataset
from churn.evaluation.metrics import (
    find_best_threshold_by_f1, evaluate_on_test,
    plot_confusion_matrix, plot_pr_curve, plot_feature_importance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    val_df = pd.read_parquet(settings.data_processed_dir / "val.parquet")
    test_df = pd.read_parquet(settings.data_processed_dir / "test.parquet")

    model = load_model(settings.project_root / "models" / "lgbm_churn.txt")

    X_val, y_val = _make_dataset(val_df)
    val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    threshold = find_best_threshold_by_f1(y_val, val_proba)
    logger.info("Selected threshold (from val, F1-optimal): %.4f", threshold)
    
    metadata = {"threshold": threshold, "version": "1"}
    (settings.project_root / "models" / "metadata.json").write_text(json.dumps(metadata, indent=2))

    X_test, y_test = _make_dataset(test_df)
    test_metrics = evaluate_on_test(model, X_test, y_test, threshold)

    out_dir = settings.project_root / "reports"
    out_dir.mkdir(exist_ok=True)

    test_proba = model.predict(X_test, num_iteration=model.best_iteration)
    test_pred = (test_proba >= threshold).astype(int)

    plot_confusion_matrix(y_test, test_pred, out_dir / "confusion_matrix.png")
    plot_pr_curve(y_test, test_proba, out_dir / "pr_curve.png")
    plot_feature_importance(model, out_dir / "feature_importance.png")

    with mlflow.start_run(run_name="test_evaluation"):
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifact(str(out_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(out_dir / "pr_curve.png"))
        mlflow.log_artifact(str(out_dir / "feature_importance.png"))

    logger.info("Evaluation artifacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
"""Final evaluation: metrics, plots, feature importance. Test set touched once, by design."""
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

logger = logging.getLogger(__name__)


def find_best_threshold_by_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Sweep thresholds on VALIDATION predictions only. Never call this with test data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx])


def evaluate_on_test(
    model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict:
    proba = model.predict(X_test, num_iteration=model.best_iteration)
    pred_labels = (proba >= threshold).astype(int)

    metrics = {
        "test_auc_pr": average_precision_score(y_test, proba),
        "test_auc_roc": roc_auc_score(y_test, proba),
        "test_f1": f1_score(y_test, pred_labels),
        "threshold_used": threshold,
    }
    logger.info("Test metrics: %s", metrics)
    logger.info("\n%s", classification_report(y_test, pred_labels, target_names=["No Churn", "Churn"]))
    return metrics


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(y_test: pd.Series, y_proba: np.ndarray, out_path: Path) -> None:
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = y_test.mean()

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"Model (AP={ap:.3f})")
    plt.axhline(baseline, color="gray", linestyle="--", label=f"No-skill baseline ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model: lgb.Booster, out_path: Path, top_n: int = 15) -> None:
    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=model.feature_name()
    ).sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    importance.iloc[::-1].plot(kind="barh")
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} Feature Importances (gain)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
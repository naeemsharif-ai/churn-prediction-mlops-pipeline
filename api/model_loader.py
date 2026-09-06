"""Loads model + threshold once at startup, reused across requests."""
import json
import logging
from pathlib import Path

import lightgbm as lgb

logger = logging.getLogger(__name__)


class ModelBundle:
    def __init__(self, model: lgb.Booster, threshold: float, version: str):
        self.model = model
        self.threshold = threshold
        self.version = version


def load_model_bundle(model_dir: Path) -> ModelBundle:
    model_path = model_dir / "lgbm_churn.txt"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `scripts/run_training.py` first."
        )

    model = lgb.Booster(model_file=str(model_path))

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        threshold = metadata.get("threshold", 0.5)
        version = metadata.get("version", "unknown")
    else:
        logger.warning("No metadata.json found — using default threshold 0.5")
        threshold, version = 0.5, "unknown"

    logger.info("Loaded model version=%s, threshold=%.4f", version, threshold)
    return ModelBundle(model, threshold, version)
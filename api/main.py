"""FastAPI churn prediction service."""
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.config import settings
from churn.data.clean import fix_total_charges, encode_target
from churn.features.engineer import engineer_features
from api.model_loader import ModelBundle, load_model_bundle
from api.schemas import (
    PredictionRequest, PredictionResponse, PredictionResult, HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_dir = settings.project_root / "models"
    model_bundle = load_model_bundle(model_dir)
    yield
    model_bundle = None


app = FastAPI(
    title="Churn Prediction API",
    description="Serves churn probability predictions for telecom customers.",
    version="1.0.0",
    lifespan=lifespan,
)


def _records_to_features(records: list[dict]) -> pd.DataFrame:
    """Mirrors the Stage 4 pipeline exactly, minus the raw-CSV-specific quirks
    (TotalCharges here arrives as float already, per the API schema — the
    string-blank issue is a raw-CSV artifact that doesn't exist at inference time)."""
    df = pd.DataFrame(records)
    df = engineer_features(df)
    return df


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_bundle is not None,
        model_version=model_bundle.version if model_bundle else "none",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        records = [r.model_dump() for r in request.records]
        X = _records_to_features(records)
        probas = model_bundle.model.predict(X, num_iteration=model_bundle.model.best_iteration)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    predictions = [
        PredictionResult(
            churn_probability=float(p),
            churn_prediction=bool(p >= model_bundle.threshold),
            threshold_used=model_bundle.threshold,
        )
        for p in probas
    ]
    return PredictionResponse(predictions=predictions, model_version=model_bundle.version)
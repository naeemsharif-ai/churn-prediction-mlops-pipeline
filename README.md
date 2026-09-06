# Customer Churn Prediction — End-to-End ML Pipeline

Production-style ML pipeline for predicting telecom customer churn: schema-validated
data ingestion, feature engineering, hyperparameter-tuned LightGBM model, MLflow
experiment tracking, and a containerized FastAPI serving layer.

## Problem & Approach

Binary classification on the [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)
(~7K customers, 26.5% churn rate). Given the class imbalance, the pipeline optimizes
for **AUC-PR** rather than AUC-ROC, since ROC-AUC overstates performance on imbalanced
targets and doesn't reflect precision on the minority (churn) class the business
actually cares about.

## Results

| Metric | Validation | Test |
|---|---|---|
| AUC-PR | 0.666 | 0.613 |
| AUC-ROC | 0.842 | 0.830 |
| F1 (threshold=0.306, selected on val) | 0.520 | 0.604 |

Classification threshold selected by maximizing F1 on the validation set only — the
test set is touched exactly once, at final evaluation.

Top predictive features (by LightGBM gain): `Contract`, `tenure`,
`avg_historical_monthly_charge`, `InternetService` — consistent with EDA findings
(month-to-month contracts and low tenure both show visibly elevated churn rates).

---

## Running Locally

### Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- macOS only: `brew install libomp` (LightGBM requires OpenMP)

### 1. Clone and install dependencies

```bash
git clone <your-fork-url> churn-pipeline
cd churn-pipeline
uv sync
```

This installs the exact environment from `uv.lock`.

### 2. Build the dataset

```bash
uv run python -m churn.data.pipeline
```

Downloads the raw Telco CSV → validates schema (pandera) → cleans → engineers
features → writes stratified train/val/test splits.

**Verify:**
```bash
uv run python -c "
import pandas as pd
from churn.config import settings
for name in ['train', 'val', 'test']:
    df = pd.read_parquet(settings.data_processed_dir / f'{name}.parquet')
    print(name, df.shape, df['Churn'].mean().round(4))
"
```
Expected: `train (5633, 24) 0.2654`, `val (1057, 24) 0.2649`, `test (353, 24) 0.2663`.

### 3. Train the model

```bash
uv run python scripts/run_training.py
```

Runs a 30-trial Optuna hyperparameter search with 5-fold stratified CV, refits with
early stopping, logs everything to MLflow, and registers the model as `churn_lgbm`.
**Takes roughly 8 minutes.**

**Verify:** `models/lgbm_churn.txt` is created; CV AUC-PR and val AUC-PR should both
land around 0.65–0.68.

### 4. Evaluate on the held-out test set

```bash
uv run python scripts/run_evaluation.py
```

Selects the F1-optimal classification threshold on validation data (never on test),
evaluates once on the test set, and writes `models/metadata.json` plus diagnostic
plots.

**Verify:** test AUC-PR ≈ 0.60–0.65, AUC-ROC ≈ 0.80–0.87; `reports/confusion_matrix.png`,
`reports/pr_curve.png`, `reports/feature_importance.png` are created.

> **Note:** Step 6 (serving) requires this step to have run at least once — the API
> reads `models/metadata.json` for its decision threshold and falls back to 0.5 with
> a warning if it's missing.

### 5. (Optional) Inspect experiments in the MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://localhost:5000](http://localhost:5000). You should see a `churn-prediction`
experiment with an `hpo_search` parent run (30 nested trial runs), a `final_model` run,
and a `test_evaluation` run, plus `churn_lgbm` under the Model Registry.

### 6. Serve predictions

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

```bash
curl http://localhost:8000/health
```
Expected: `{"status":"ok","model_loaded":true,"model_version":"1"}`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
      "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
      "MultipleLines": "No", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "OnlineBackup": "Yes",
      "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "Yes", "StreamingMovies": "No",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
      "TotalCharges": 427.5
    }]
  }'
```

This is a high-risk profile (month-to-month, fiber, low tenure) — expect a
relatively high `churn_probability`.

### 7. Run the test suite

```bash
uv run pytest -v
```

Expected: 5 tests passing (2 data/feature unit tests, 3 API integration tests).

### 8. (Optional, requires Docker) Containerized run

```bash
docker compose build
docker compose up -d
curl http://localhost:8000/health
docker compose logs api --tail 50
```

`models/lgbm_churn.txt` and `models/metadata.json` must already exist locally
(steps 3–4) before building — the Dockerfile copies them into the image rather than
training inside the container.

---

## Order of Operations

Steps 2 → 3 → 4 are strictly sequential and each depends on the previous one's
output. Step 6 (serving) additionally depends on step 4 for the decision threshold.
Steps 5, 7, and 8 can run any time after their respective dependencies are met.

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| `FileNotFoundError: models/lgbm_churn.txt` | Step 3 (training) was skipped |
| API logs "No metadata.json found — using default threshold 0.5" | Step 4 (evaluation) was skipped |
| `ModuleNotFoundError: churn` or `api` | Ran `python` directly instead of `uv run python` |
| Raw data download returns 404 | Upstream GitHub mirror moved — update the URL in `src/churn/data/load.py` |
| `Address already in use` on port 8000/5000 | A previous `uvicorn` or `mlflow ui` process is still running |
| Metrics differ from the table above by more than ~±0.02 | Normal — Optuna/LightGBM threading introduces minor run-to-run variance |

## Project Structure

```
churn-pipeline/
├── src/churn/            # reusable library code (data, features, models)
├── api/                  # FastAPI serving layer
├── scripts/               # one-off entry points (training, evaluation)
├── notebooks/             # exploratory EDA only — nothing production imports from here
├── tests/                 # pytest unit + API tests
├── data/                  # raw + processed data (gitignored)
├── models/                # trained model + metadata (gitignored or committed — your call)
├── reports/                # evaluation plots
├── mlruns/ / mlflow.db     # MLflow tracking store (gitignored)
├── Dockerfile / docker-compose.yml
└── pyproject.toml / uv.lock
```

## Tech Stack

Python 3.11 · uv · Pandas · pandera · LightGBM · Optuna · MLflow · FastAPI · Docker

## Known Limitations

- Single-snapshot dataset — no temporal validation; `tenure` is treated as a
  point-in-time feature rather than a time series.
- The F1-optimal threshold is a simplification; a real deployment would derive it
  from actual retention-offer cost vs. customer lifetime value.
- Feature importance (gain-based) indicates predictive usefulness, not causation.
- Docker image build and AWS deployment are documented (see architecture notes) but
  not executed in this repo, due to no local Docker access during development. CI/CD
  via GitHub Actions would build the image on a hosted runner instead.
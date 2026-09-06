import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_valid_input(client):
    payload = {
        "records": [{
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
            "TotalCharges": 427.5,
        }]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    assert 0.0 <= body["predictions"][0]["churn_probability"] <= 1.0


def test_predict_endpoint_rejects_invalid_contract(client):
    payload = {
        "records": [{
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Not a real contract type", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
            "TotalCharges": 427.5,
        }]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
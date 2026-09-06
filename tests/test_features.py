# tests/test_features.py
import pandas as pd
from churn.features.engineer import add_derived_features

def test_num_addon_services_counts_correctly():
    df = pd.DataFrame({
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["Yes", "No internet service"],
        "TotalCharges": [100.0, 50.0],
        "tenure": [10, 5],
        "MonthlyCharges": [10.0, 10.0],
    })
    result = add_derived_features(df)
    assert result["num_addon_services"].tolist() == [3, 0]
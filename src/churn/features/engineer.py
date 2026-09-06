"""Feature enginering. Pure functions, no I/O - reused identically at train and inference time."""
import pandas as pd

BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

# columns with Yes/No/"No internet service" or "No phone service" - treated as
# 3-level categoricals, not collapsed to binary, since "No X service" carries
# distinct signal from "has X but declined" in the churn-rate EDA
SERVICE_DEPENDENT_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLS = [
    "gender", "InternetService", "Contract", "PaymentMethod",
    *SERVICE_DEPENDENT_COLS,
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

def encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features with clear domain rationale - avoid throwing in
    arbitrary interactions without a story you can defend in an interview."""
    df = df.copy()

    # avg monthly spend vs current rate - a customer paying much more/less than
    # their current MonthlyCharges historically may signal a recent plan change
    df["avg_historical_monthly_charge"] = (
        df["TotalCharges"] / df["tenure"].replace(0, 1)
    )
    df["charge_deviation_from_avg"] = (
        df["MonthlyCharges"] - df["avg_historical_monthly_charge"]
    )

    # count of subscribed add-on services - proxy for account "stickiness"
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    # tenure bucket - captures non-linear early-tenure churn risk that
    # tree models can already find, but makes the split human-interpretable
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    )

    return df

def set_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM/CatBoost both accept pandas 'category' dtype natively -
    no need for manual one-hot or label encoding."""
    df = df.copy()
    for col in CATEGORICAL_COLS + ["tenure_bucket"]:
        df[col] = df[col].astype("category")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_binary_yes_no(df)
    df = add_derived_features(df)
    df = set_categorical_dtypes(df)
    return df
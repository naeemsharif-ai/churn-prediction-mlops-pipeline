"""Deterministic cleaning transformations. Pure functions only - no I/O side effects."""
import pandas as pd

DROP_COLS = ["customerID"]

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges ships as string with blank entries for tenure==0 customers.
    These are structural zeros (no billing history yet), not missing data."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df

def drop_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLS)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full deterministic cleaning pipeline, in fixed order."""
    df = fix_total_charges(df)
    df = encode_target(df)
    df = drop_identifier_cols(df)
    return df
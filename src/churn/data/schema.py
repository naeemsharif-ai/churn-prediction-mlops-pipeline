"""Declarative schema contract for the raw Telco churn dataset."""
import pandera.pandas as pa
from pandera.typing import Series

class TelcoChurnRawSchema(pa.DataFrameModel):
    customerID: Series[str] = pa.Field(unique=True)
    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0,1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0, le=100)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(nullable=False)
    InternetService: Series[str] = pa.Field(nullable=False)
    OnlineSecurity: Series[str] = pa.Field(nullable=False)
    OnlineBackup: Series[str] = pa.Field(nullable=False)
    DeviceProtection: Series[str] = pa.Field(nullable=False)
    TechSupport: Series[str] = pa.Field(nullable=False)
    StreamingTV: Series[str] = pa.Field(nullable=False)
    StreamingMovies: Series[str] = pa.Field(nullable=False)
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(nullable=False)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[str] = pa.Field(nullable=False) # raw has this as string w/ blanks - caught in cleaning
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        strict = True   # fail on unexpected columns
        coerce = True
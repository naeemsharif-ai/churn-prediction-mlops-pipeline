# tests/test_data.py
import pandas as pd
from churn.data.clean import fix_total_charges

def test_fix_total_charges_handles_blank_strings():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "108.15"]})
    result = fix_total_charges(df)
    assert result["TotalCharges"].tolist() == [29.85, 0.0, 108.15]
    assert result["TotalCharges"].dtype == float
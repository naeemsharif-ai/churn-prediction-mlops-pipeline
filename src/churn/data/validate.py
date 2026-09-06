"""Validate raw data against the declared schema."""
import logging

import pandas as pd
from pandera.errors import SchemaError

from churn.data.schema import TelcoChurnRawSchema

logger = logging.getLogger(__name__)

def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return TelcoChurnRawSchema.validate(df, lazy=True)
    except SchemaError as exc:
        logger.error("Schema validation failed:\n%s", exc)
        raise
import pandas as pd
from typing import Dict, Union
import logging

logger = logging.getLogger("MoneyMap.Schema")

TRANSACTION_SCHEMA = {
    "transaction_id": str,
    "timestamp": "datetime64[ns]",
    "sender_id": str,
    "receiver_id": str,
    "amount": float,
    "currency": str,
    "channel": str,
    "location": str,
}

def validate_schema(df: pd.DataFrame, schema: Dict[str, Union[type, str]]) -> pd.DataFrame:
    missing = [col for col in schema if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")
    
    for col, col_type in schema.items():
        if col_type == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(col_type, errors="ignore")
    
    logger.info("Schema validation passed.")
    return df

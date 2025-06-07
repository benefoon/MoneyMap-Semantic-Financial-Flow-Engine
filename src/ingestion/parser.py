import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger("MoneyMap.Parser")

def standardize_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns=column_map)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    logger.info(f"Standardized columns: {list(df.columns)}")
    return df

def drop_useless_columns(df: pd.DataFrame, threshold_null: float = 0.95) -> pd.DataFrame:
    null_ratios = df.isnull().mean()
    drop_cols = null_ratios[null_ratios > threshold_null].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    logger.info(f"Dropped columns due to null threshold: {drop_cols}")
    return df

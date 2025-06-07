import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger("MoneyMap.FeatureMapper")

def encode_categoricals(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"Encoded column: {col}")
    return df, encoders

def normalize_numerics(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"Normalized columns: {columns}")
    return df, scaler

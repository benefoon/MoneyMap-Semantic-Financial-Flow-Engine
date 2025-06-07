import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger("MoneyMap.Loader")

def load_csv(path: Union[str, Path], date_col: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading CSV from: {path}")
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    return df

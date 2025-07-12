import pandas as pd
from src.config.paths import RAW_TRANSACTIONS
from src.utils.logger import get_logger
from src.config.paths import LOGS_DIR

logger = get_logger("data_loader", log_path=LOGS_DIR)

def load_transactions(path: str = None) -> pd.DataFrame:
    source = path if path else RAW_TRANSACTIONS
    try:
        df = pd.read_csv(source)
        logger.info(f"Loaded data from {source} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {source}: {e}")
        raise

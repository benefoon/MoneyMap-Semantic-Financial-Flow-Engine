from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RAW_TRANSACTIONS = RAW_DATA / "transactions.csv"
CLEAN_TRANSACTIONS = PROCESSED_DATA / "transactions_cleaned.csv"
NODE_EMBEDDINGS = PROCESSED_DATA / "node_embeddings.npy"
GRAPH_PICKLE = PROCESSED_DATA / "financial_graph.gpickle"

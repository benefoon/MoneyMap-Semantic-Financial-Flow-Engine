import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger("MoneyMap.Synthesizer")

def generate_synthetic_transactions(n: int = 10000) -> pd.DataFrame:
    logger.info(f"Generating {n} synthetic transactions...")
    users = [f"U{i:05d}" for i in range(500)]
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=random.randint(0, 525600)) for _ in range(n)]
    
    data = {
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "timestamp": timestamps,
        "sender_id": np.random.choice(users, n),
        "receiver_id": np.random.choice(users, n),
        "amount": np.random.exponential(scale=150.0, size=n).round(2),
        "currency": np.random.choice(["USD", "EUR", "GBP", "BTC"], n),
        "channel": np.random.choice(["wire", "card", "app", "swift"], n),
        "location": np.random.choice(["NY", "LDN", "BER", "TKY", "SGP"], n)
    }

    return pd.DataFrame(data)

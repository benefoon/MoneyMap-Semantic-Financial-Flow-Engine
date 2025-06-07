import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Optional
import logging

logger = logging.getLogger("MoneyMap.GraphBuilder")

def transactions_to_graph(df: pd.DataFrame, features: Optional[List[str]] = None) -> Data:
    logger.info("Converting transaction DataFrame to PyG graph...")
    
    node_ids = pd.Index(df["sender_id"].tolist() + df["receiver_id"].tolist()).unique()
    node_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    edge_index_np = df[["sender_id", "receiver_id"]].applymap(lambda x: node_map[x]).to_numpy().T
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    
    if features:
        node_feat_df = df[["sender_id"] + features].groupby("sender_id").mean().reindex(node_ids).fillna(0)
        node_features = torch.tensor(node_feat_df.to_numpy(), dtype=torch.float)
    else:
        node_features = torch.ones((len(node_ids), 1), dtype=torch.float)
    
    logger.info(f"Constructed graph with {len(node_ids)} nodes and {df.shape[0]} edges.")
    return Data(x=node_features, edge_index=edge_index)

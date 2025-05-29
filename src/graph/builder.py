# src/graph/builder.py

import networkx as nx
import pandas as pd
from typing import Optional

class TransactionGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_transaction(self, sender: str, receiver: str, amount: float, timestamp: str, embedding=None):
        if sender not in self.graph:
            self.graph.add_node(sender, entity_type='account')
        if receiver not in self.graph:
            self.graph.add_node(receiver, entity_type='account')

        edge_attrs = {'amount': amount, 'timestamp': timestamp}
        if embedding is not None:
            edge_attrs['embedding'] = embedding.detach().cpu().numpy()

        self.graph.add_edge(sender, receiver, **edge_attrs)

    def build_from_dataframe(self, df: pd.DataFrame, embedding_col: Optional[str] = None):
        for idx, row in df.iterrows():
            emb = row[embedding_col] if embedding_col else None
            self.add_transaction(
                sender=row['sender_account'],
                receiver=row['receiver_account'],
                amount=row['amount'],
                timestamp=row['timestamp'],
                embedding=emb
            )

    def save_graph(self, path: str):
        nx.write_gpickle(self.graph, path)

    def load_graph(self, path: str):
        self.graph = nx.read_gpickle(path)

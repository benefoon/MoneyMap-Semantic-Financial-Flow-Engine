import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Optional
import numpy as np

class GNNPreprocessor:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def to_pyg_data(self, use_embeddings: bool = True) -> Data:
        node_map = {node: i for i, node in enumerate(self.graph.nodes)}
        edge_index = []
        edge_attr = []

        for u, v, attrs in self.graph.edges(data=True):
            edge_index.append([node_map[u], node_map[v]])
            if use_embeddings and 'embedding' in attrs:
                edge_attr.append(attrs['embedding'])
            else:
                edge_attr.append([attrs.get('amount', 0.0)])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        x = torch.ones((len(self.graph.nodes), 1), dtype=torch.float)  # Dummy node features
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

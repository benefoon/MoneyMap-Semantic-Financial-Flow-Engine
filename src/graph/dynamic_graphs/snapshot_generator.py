import networkx as nx
import pandas as pd
from typing import List
from datetime import timedelta

class SnapshotGenerator:
    def __init__(self, full_graph: nx.DiGraph):
        self.full_graph = full_graph

    def generate_time_slices(self, interval: str = '30D') -> List[nx.DiGraph]:
        timestamps = [data['timestamp'] for _, _, data in self.full_graph.edges(data=True)]
        min_time, max_time = min(timestamps), max(timestamps)
        current_time = min_time
        snapshots = []

        while current_time < max_time:
            end_time = current_time + pd.Timedelta(interval)
            subgraph = nx.DiGraph()
            for u, v, attrs in self.full_graph.edges(data=True):
                if current_time <= attrs['timestamp'] < end_time:
                    subgraph.add_edge(u, v, **attrs)
            snapshots.append(subgraph)
            current_time = end_time

        return snapshots

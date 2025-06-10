import networkx as nx
import community as community_louvain

def detect_communities(graph: nx.Graph) -> dict:
    """
    Detect communities in the transaction graph using the Louvain method.
    
    Args:
        graph (nx.Graph): The transaction graph.
    
    Returns:
        dict: Mapping from node to community id.
    """
    partition = community_louvain.best_partition(graph)
    return partition

import networkx as nx
from node2vec import Node2Vec
import numpy as np
import os
import logging

class Node2VecEmbedder:
    """
    Node2Vec embedding generator for transaction graphs.
    Converts a NetworkX graph into a dense embedding matrix for nodes.
    """

    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4,
        window: int = 5,
        min_count: int = 1,
        batch_words: int = 4,
        seed: int = 42,
        logger: logging.Logger = None
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.embedding_matrix = None

    def fit(self, graph: nx.Graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input must be a NetworkX Graph instance.")

        self.logger.info("Initializing Node2Vec model...")
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            seed=self.seed
        )

        self.logger.info("Training Node2Vec model...")
        self.model = node2vec.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words
        )

        self.logger.info("Embedding training completed.")
        self.embedding_matrix = self._build_embedding_matrix(graph)

    def _build_embedding_matrix(self, graph: nx.Graph):
        emb_matrix = {}
        for node in graph.nodes:
            try:
                emb = self.model.wv[str(node)]
                emb_matrix[node] = np.array(emb)
            except KeyError:
                emb_matrix[node] = np.zeros(self.dimensions)
                self.logger.warning(f"Node {node} not found in Word2Vec vocabulary.")

        return emb_matrix

    def get_embedding(self, node_id):
        return self.embedding_matrix.get(node_id, None)

    def save(self, path: str):
        if not self.model:
            raise RuntimeError("Model has not been trained yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.wv.save_word2vec_format(path)
        self.logger.info(f"Embeddings saved to {path}")

    def load(self, path: str):
        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(path)
        self.logger.info(f"Embeddings loaded from {path}")

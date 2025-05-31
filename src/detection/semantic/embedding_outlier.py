import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingOutlierDetector:
    """
    Detects semantic outliers by analyzing distances between node embeddings.
    """

    def __init__(self, method='cosine', threshold=0.8):
        self.method = method
        self.threshold = threshold

    def fit(self, embeddings):
        self.embeddings = embeddings
        self.sim_matrix = cosine_similarity(embeddings)

    def score(self):
        sim_mean = np.mean(self.sim_matrix, axis=1)
        return 1 - sim_mean  # lower similarity => higher anomaly

    def detect(self):
        scores = self.score()
        return np.where(scores > self.threshold)[0]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class DominantGCNAutoEncoder(nn.Module):
    """
    DOMINANT: Deep autoencoder with GCN layers for unsupervised graph anomaly detection.
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(DominantGCNAutoEncoder, self).__init__()
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, latent_dim)
        self.decoder1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        z = F.relu(self.encoder1(x, edge_index))
        z = self.encoder2(z, edge_index)
        h = F.relu(self.decoder1(z))
        x_hat = self.decoder2(h)
        return x_hat, z

    def compute_anomaly_score(self, x, x_hat):
        """
        Compute reconstruction-based anomaly score (MSE per node).
        """
        score = torch.mean((x - x_hat) ** 2, dim=1)
        return score

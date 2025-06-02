import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, output_dim)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        return self.lin2(h)

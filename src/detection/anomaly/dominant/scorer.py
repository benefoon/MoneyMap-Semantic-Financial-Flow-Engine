import torch

def reconstruction_error(x_true, x_reconstructed):
    return torch.mean((x_true - x_reconstructed) ** 2, dim=1)

def rank_anomalies(scores, top_k=10):
    return torch.topk(scores, top_k).indices.tolist()

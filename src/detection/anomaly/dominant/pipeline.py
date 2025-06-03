import torch
from .model import DominantAutoencoder
from .trainer import train_model
from .scorer import reconstruction_error, rank_anomalies
from .utils import normalize_features
from .config import DOMINANT_CONFIG

def run_dominant(data, raw_features):
    x = normalize_features(raw_features)
    data.x = x

    model = DominantAutoencoder(
        input_dim=x.shape[1],
        hidden_dim=DOMINANT_CONFIG["hidden_dim"],
        latent_dim=DOMINANT_CONFIG["latent_dim"]
    )

    trained_model = train_model(model, data, DOMINANT_CONFIG)

    with torch.no_grad():
        trained_model.eval()
        x_hat, _ = trained_model(data.x.to(DOMINANT_CONFIG["device"]), data.edge_index.to(DOMINANT_CONFIG["device"]))
        scores = reconstruction_error(data.x.to(DOMINANT_CONFIG["device"]), x_hat)

    anomalies = rank_anomalies(scores, top_k=15)
    return scores.cpu().numpy(), anomalies

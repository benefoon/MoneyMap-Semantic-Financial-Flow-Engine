DOMINANT_CONFIG = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

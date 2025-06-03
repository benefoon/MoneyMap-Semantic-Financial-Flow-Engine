from sklearn.preprocessing import StandardScaler
import torch

def normalize_features(x_np):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_np)
    return torch.tensor(x_scaled, dtype=torch.float32)

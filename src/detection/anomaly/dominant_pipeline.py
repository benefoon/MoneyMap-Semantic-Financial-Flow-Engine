import torch
from src.detection.anomaly.dominant import DominantGCNAutoEncoder
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def run_dominant_pipeline(node_features, edge_index, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    x_np = scaler.fit_transform(node_features)
    x = torch.tensor(x_np, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    model = DominantGCNAutoEncoder(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, _ = model(x, edge_index)
        loss = torch.mean((x - x_hat) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_hat, _ = model(x, edge_index)
        scores = model.compute_anomaly_score(x, x_hat)

    return scores.cpu().numpy()

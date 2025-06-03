import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .scorer import reconstruction_error

def train_model(model, data, config):
    model.to(config["device"])
    x, edge_index = data.x.to(config["device"]), data.edge_index.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    best_loss = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(config["epochs"]), desc="Training DOMINANT"):
        model.train()
        optimizer.zero_grad()
        x_hat, _ = model(x, edge_index)
        loss = torch.mean((x - x_hat) ** 2)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    return model

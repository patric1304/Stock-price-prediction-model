import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import StockDataset
from src.model import StockPredictor
from src.preprocessing import scale_features

def train_model(X, y, epochs=20, batch_size=32, lr=1e-3):
    # scale
    X_scaled, y_scaled, _, _ = scale_features(X, y)
    dataset = StockDataset(X_scaled, y_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = X.shape[1]
    model = StockPredictor(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_batch = y_batch.view(-1, 1)  # fix shape
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return model

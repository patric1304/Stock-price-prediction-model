import torch
from torch.utils.data import DataLoader
from src.model import StockPredictor
from src.dataset import StockDataset

def train_model(X_train, y_train, lr=1e-3, epochs=50, batch_size=32):
    dataset = StockDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = StockPredictor(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    return model

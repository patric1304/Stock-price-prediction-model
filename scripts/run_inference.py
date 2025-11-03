import torch
from src.model import StockPredictor
import numpy as np
from pathlib import Path

MODEL_PATH = Path("data/processed/stock_model.pth")

# load model
input_dim = 106  # adjust if different
model = StockPredictor(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# dummy input
X_new = np.random.rand(1, input_dim).astype(np.float32)
X_tensor = torch.tensor(X_new)
y_pred = model(X_tensor).item()
print(f"Predicted next-day stock price: {y_pred:.2f}")

import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # single layer linear regression

    def forward(self, x):
        return self.linear(x)

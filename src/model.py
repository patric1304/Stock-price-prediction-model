import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Linear regression style
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

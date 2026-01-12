import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import HISTORY_DAYS

class AdvancedStockPredictor(nn.Module):
    """
    Advanced deep learning model for stock price prediction.
    
    Architecture Components:
    - LSTM/GRU layers for temporal pattern recognition
    - Multi-head attention mechanism for feature importance
    - Residual connections to prevent vanishing gradients
    - Multiple dropout layers for regularization
    - Batch normalization for stable training
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, history_days: int = HISTORY_DAYS):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.history_days = int(history_days)

                                                                               
        self.price_features_per_day = 5
        self.window_dim = self.history_days * self.price_features_per_day
        self.extra_dim = max(0, self.input_dim - self.window_dim)
        self.per_step_input_dim = self.price_features_per_day + self.extra_dim
        
                                               
        self.input_projection = nn.Sequential(
            nn.Linear(self.per_step_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
                                                                       
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
                                                    
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,                      
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
                                                     
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim * 2, dropout) for _ in range(3)
        ])
        
                                                            
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 4, 1)
        )
        
                            
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier/Glorot initialization for better gradient flow"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x):
                                                                                 
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input [batch, features], got shape {tuple(x.shape)}")
        if x.size(1) < self.window_dim:
            raise ValueError(
                f"Input feature dim {x.size(1)} is smaller than expected window dim {self.window_dim}. "
                "Ensure gather_data produces HISTORY_DAYS*5 OHLCV features first."
            )

        window_flat = x[:, :self.window_dim]
        window_seq = window_flat.view(x.size(0), self.history_days, self.price_features_per_day)

        if self.extra_dim > 0:
            extra = x[:, self.window_dim:self.window_dim + self.extra_dim]
            extra_seq = extra.unsqueeze(1).expand(-1, self.history_days, -1)
            seq = torch.cat([window_seq, extra_seq], dim=-1)
        else:
            seq = window_seq

                                                                    
        x = self.input_projection(seq)
        
                         
        lstm_out, _ = self.lstm(x)                                       
        
                                  
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)                                       
                                                                        
        attn_out = attn_out[:, -1, :]                         
        
                         
        out = attn_out
        for block in self.residual_blocks:
            out = block(out)
        
                          
        prediction = self.output_network(out)
        
        return prediction


class ResidualBlock(nn.Module):
    """Residual block with skip connection for deeper networks"""
    
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual                   
        out = F.relu(out)
        out = self.dropout(out)
        return out


class StockPredictor(nn.Module):
    """Legacy simple model - kept for backward compatibility"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

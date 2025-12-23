import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal dependencies (assuming sequential features)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional LSTM
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Deep residual network with skip connections
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim * 2, dropout) for _ in range(3)
        ])
        
        # Output layers with progressive dimension reduction
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
        
        # Initialize weights
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
        # Input projection
        x = self.input_projection(x)  # [batch, hidden_dim]
        
        # Add sequence dimension for LSTM (treating features as sequence)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, 1, hidden_dim*2]
        
        # Self-attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [batch, 1, hidden_dim*2]
        attn_out = attn_out.squeeze(1)  # [batch, hidden_dim*2]
        
        # Residual blocks
        out = attn_out
        for block in self.residual_blocks:
            out = block(out)
        
        # Final prediction
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
        out = out + residual  # Skip connection
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

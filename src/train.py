import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.dataset import StockDataset
from src.model import AdvancedStockPredictor, StockPredictor
from src.preprocessing import scale_features


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_model_advanced(
    X, y, 
    epochs=200, 
    batch_size=64, 
    lr=1e-3,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    train_split=0.7,
    val_split=0.15,
    patience=15,
    target_mode=None,
    use_gpu=True,
    checkpoint_dir='data/checkpoints',
    verbose=True
):
    """
    Advanced training with proper train/val/test split and regularization.
    
    Args:
        X: Feature matrix
        y: Target values
        epochs: Maximum training epochs
        batch_size: Batch size for training
        lr: Learning rate
        hidden_dim: Hidden dimension for the model
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        train_split: Proportion of data for training (0.7 = 70%)
        val_split: Proportion of data for validation (0.15 = 15%)
        patience: Early stopping patience
        use_gpu: Use GPU if available
        checkpoint_dir: Directory to save checkpoints
        verbose: Print training progress
    
    Returns:
        model: Trained model
        history: Training history
        scalers: Tuple of (X_scaler, y_scaler)
        test_metrics: Dictionary of test set performance
    """
    
    # Set device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Proper train/val/test split
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # First split: separate test set (remaining after train+val)
    test_size = 1.0 - train_split - val_split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_split / (train_split + val_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted, random_state=42, shuffle=False
    )
    
    if verbose:
        print(f"\nData Split:")
        print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
    
    # Scale features using only training data to prevent data leakage
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Fit scalers on training data only
    X_train_scaled, y_train_scaled, scaler_X, scaler_y = scale_features(X_train, y_train)
    
    # Transform validation and test data using training scalers
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Create datasets
    train_dataset = StockDataset(X_train_scaled, y_train_scaled)
    val_dataset = StockDataset(X_val_scaled, y_val_scaled)
    test_dataset = StockDataset(X_test_scaled, y_test_scaled)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X.shape[1]
    model = AdvancedStockPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5  # L2 regularization
    )
    
    # Learning rate scheduler (verbose removed - not supported in all PyTorch versions)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    if verbose:
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_dataset)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'target_mode': target_mode,
                'model_kwargs': {
                    'input_dim': int(input_dim),
                    'hidden_dim': int(hidden_dim),
                    'num_layers': int(num_layers),
                    'dropout': float(dropout),
                    'history_days': int(getattr(model, 'history_days', 0) or 0),
                },
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            early_stopping.load_best_model(model)
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            
            all_predictions.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    test_loss /= len(test_dataset)
    
    # Calculate additional metrics
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Inverse transform to original scale
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    mse = np.mean((predictions_original - targets_original) ** 2)
    mae = np.mean(np.abs(predictions_original - targets_original))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets_original - predictions_original) / targets_original)) * 100
    
    test_metrics = {
        'test_loss': test_loss,
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Test Set Performance")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': float(best_val_loss),
        'test_metrics': test_metrics,
        'target_mode': target_mode,
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'training_config': {
            'batch_size': batch_size,
            'learning_rate': lr,
            'train_split': train_split,
            'val_split': val_split,
            'patience': patience
        }
    }
    
    summary_path = Path(checkpoint_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model, history, (scaler_X, scaler_y), test_metrics


def train_model(X, y, epochs=20, batch_size=32, lr=1e-3):
    """Legacy training function - kept for backward compatibility"""
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

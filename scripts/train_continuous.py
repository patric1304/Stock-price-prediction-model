import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_gathering import gather_data
from src.preprocessing import scale_features
from src.model import StockPredictor
from src.config import TRAINING_DATA_DAYS, MAX_DAILY_REQUESTS
import torch
import numpy as np
from datetime import datetime
import pickle

# Weekly rotation of S&P 500 stocks - OPTIMIZED for NewsAPI (100 requests/day, 20 days lookback)
# Each stock uses 20 requests (20 days of news)
# 100 requests ÷ 20 = 5 stocks per day max
# Using 4 stocks per day for safety buffer = 80 requests/day
STOCK_GROUPS = {
    "thursday": ["ROP", "RSG", "RTX", "RVTY"],
    "friday": ["SBAC", "SBUX", "SCHW", "SEE"],
    "saturday": ["SHW", "SJM", "SLB", "SNA"],
    "sunday": ["SO", "SPG", "SPGI", "SRE"],
    "monday": ["STE", "STZ", "SWK", "SWKS"],
    "tuesday": ["SYF", "SYK", "SYY", "T"],
    "wednesday": ["TAP", "TDY", "TECH", "TEL"],
    "thursday2": ["TER", "TFC", "TFX", "TGT"],
    "friday2": ["TJX", "TMUS", "TPR", "TRGP"],
    "saturday2": ["TRMB", "TROW", "TRV", "TSLA"],
    "sunday2": ["TSN", "TT", "TTWO", "TXN"],
    "monday2": ["TXT", "TYL", "UDR", "UL"],
    "tuesday2": ["UMPQ", "UNH", "UNP", "UPS"],
    "wednesday2": ["URI", "USB", "VEV", "VFC"]
}

# Determine which day of 2-week cycle (Day 1-14)
day_index = (datetime.now() - datetime(2025, 11, 7)).days % 14  # Starting from Nov 7
day_names = ["thursday", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
             "thursday2", "friday2", "saturday2", "sunday2", "monday2", "tuesday2", "wednesday2"]
day_key = day_names[day_index]
SP500_STOCKS = STOCK_GROUPS[day_key]

EPOCHS = 500  # Reasonable epochs to avoid overfitting
BATCH_SIZE = 32  # Mini-batch training
CHECKPOINT_EVERY = 100

print(f"Day {day_index + 1}/14: {', '.join(SP500_STOCKS)}")

# Prepare directories
models_dir = Path("data/processed/models")
checkpoints_dir = Path("data/checkpoints")
logs_dir = Path("data/processed/training_logs")
scalers_dir = Path("data/processed/scalers")
data_archive_dir = Path("data/processed/training_data")
for d in [models_dir, checkpoints_dir, logs_dir, scalers_dir, data_archive_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load previously accumulated data (if any)
accumulated_data_path = data_archive_dir / "accumulated_training_data.pkl"
if accumulated_data_path.exists():
    print("Loading previous training data...")
    with open(accumulated_data_path, "rb") as f:
        saved_data = pickle.load(f)
        all_X = saved_data['X']
        all_y = saved_data['y']
        all_stocks = saved_data['stocks']
    print(f"✓ Loaded {len(all_stocks)} stocks from previous days")
else:
    print("Starting fresh - no previous data")
    all_X = []
    all_y = []
    all_stocks = []

# Gather today's new stocks
successful_stocks = []
failed_stocks = []

for i, ticker in enumerate(SP500_STOCKS, 1):
    # Skip if already trained
    if ticker in all_stocks:
        print(f"  [{i}/{len(SP500_STOCKS)}] {ticker}... already trained, skipping")
        continue
        
    try:
        print(f"  [{i}/{len(SP500_STOCKS)}] {ticker}...", end=" ", flush=True)
        
        X, y = gather_data(ticker, days_back=TRAINING_DATA_DAYS)
        
        if len(X) < 5:
            print(f"skipped")
            failed_stocks.append(ticker)
            continue
        
        all_X.append(X)
        all_y.append(y)
        all_stocks.append(ticker)
        successful_stocks.append(ticker)
        print(f"✓ {len(X)} samples")
        
    except Exception as e:
        print(f"failed")
        failed_stocks.append(ticker)

if not all_X:
    print("No data to train on!")
    exit(1)

# Combine all accumulated data
X_combined = np.vstack(all_X)
y_combined = np.concatenate(all_y)

# Save accumulated data for next run
with open(accumulated_data_path, "wb") as f:
    pickle.dump({
        'X': all_X,
        'y': all_y,
        'stocks': all_stocks
    }, f)
print(f"✓ Saved accumulated data: {len(all_stocks)} total stocks")

print(f"\nTraining: {len(all_stocks)} total stocks, {len(X_combined)} samples, {X_combined.shape[1]} features")
if successful_stocks:
    print(f"New today: {', '.join(successful_stocks)}")

# Load existing model if available (incremental training across days)
existing_models = sorted(models_dir.glob("sp500_model_*.pth"))
X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)
input_dim = X_scaled.shape[1]

# Always create new model with current architecture
model = StockPredictor(input_dim)

if existing_models:
    print(f"Found previous model: {existing_models[-1].name}")
    try:
        # Try to load state dict if architectures match
        old_model = torch.load(existing_models[-1], weights_only=False)
        if hasattr(old_model, 'network'):  # New architecture
            model.load_state_dict(old_model.state_dict())
            print(f"✓ Loaded! Continuing training from previous days")
        else:  # Old architecture - start fresh
            print(f"Old architecture detected, starting fresh with new model")
    except Exception as e:
        print(f"Could not load ({str(e)[:50]}), starting fresh")
else:
    print("Starting fresh (Day 1)")

# Train the model
print(f"Training {EPOCHS} epochs with mini-batches (30-40 minutes)...")
import time
start_time = time.time()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)

# Create DataLoader for mini-batch training
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

training_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    # Mini-batch training
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    training_losses.append(avg_loss)
    
    if (epoch + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{EPOCHS}: Loss {avg_loss:.4f} | Time: {elapsed/60:.1f}min")
    
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

total_time = time.time() - start_time

# Calculate final accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    final_loss = criterion(predictions, y_tensor).item()
    
    # Reshape predictions and targets to 2D for scaler (remove extra dimensions)
    predictions_2d = predictions.numpy().reshape(-1, 1)
    y_2d = y_tensor.numpy().reshape(-1, 1)
    
    predictions_actual = scaler_y.inverse_transform(predictions_2d)
    y_actual = scaler_y.inverse_transform(y_2d)
    
    mae = np.abs(predictions_actual - y_actual).mean()
    mape = np.abs((predictions_actual - y_actual) / y_actual).mean() * 100
    rmse = np.sqrt(np.mean((predictions_actual - y_actual) ** 2))
    
    print(f"\nTraining time: {total_time/60:.1f} minutes")
    print(f"MAE: ${mae:.2f} | MAPE: {mape:.1f}% | RMSE: ${rmse:.2f}")

# Save final model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = models_dir / f"sp500_model_{timestamp}.pth"
torch.save(model, model_path)

# Save scalers
with open(scalers_dir / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(scalers_dir / "scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)


print(f"Saved: {model_path.name}")

# Save training log
log_path = logs_dir / f"training_{timestamp}.txt"
with open(log_path, "w") as f:
    f.write(f"Day {day_index + 1}/14 - {day_key.capitalize()}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total stocks trained: {len(all_stocks)}\n")
    f.write(f"New stocks today: {', '.join(successful_stocks) if successful_stocks else 'None (all already trained)'}\n")
    f.write(f"All stocks: {', '.join(all_stocks)}\n\n")
    f.write(f"Total samples: {len(X_combined):,}\n")
    f.write(f"Features: {X_combined.shape[1]}\n\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Final Loss: {final_loss:.6f}\n\n")
    f.write(f"MAE: ${mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write(f"RMSE: ${rmse:.2f}\n\n")
    f.write(f"Model: {model_path.name}\n")
    if failed_stocks:
        f.write(f"\nFailed: {', '.join(failed_stocks)}\n")

print(f"Done! Day {day_index + 1}/14 complete - Total: {len(all_stocks)} stocks")

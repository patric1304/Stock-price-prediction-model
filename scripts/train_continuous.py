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
    "thursday": ["AAPL", "MSFT", "GOOGL", "AMZN"],           # Mega-cap tech (Day 1)
    "friday": ["NVDA", "META", "TSLA", "BRK.B"],            # Growth + value leaders
    "saturday": ["UNH", "XOM", "JNJ", "JPM"],               # Healthcare + Energy + Finance
    "sunday": ["V", "PG", "MA", "HD"],                      # Payments + Consumer staples
    "monday": ["CVX", "MRK", "ABBV", "KO"],                 # Energy + Pharma + Beverages
    "tuesday": ["PEP", "AVGO", "COST", "WMT"],              # Consumer + Tech + Retail
    "wednesday": ["MCD", "CSCO", "ACN", "TMO"],             # Food + Tech + Services
    "thursday2": ["NFLX", "ABT", "CRM", "ORCL"],            # Media + Healthcare + Software
    "friday2": ["NKE", "INTC", "VZ", "CMCSA"],              # Apparel + Tech + Telecom
    "saturday2": ["AMD", "QCOM", "PM", "LLY"],              # Semiconductors + Tobacco + Pharma
    "sunday2": ["ADBE", "DHR", "TXN", "NEE"],               # Software + Industrials + Utilities
    "monday2": ["UNP", "RTX", "INTU", "HON"],               # Rail + Aerospace + Software + Industrials
    "tuesday2": ["CAT", "LOW", "BA", "GS"],                 # Construction + Retail + Aerospace + Banking
    "wednesday2": ["SPGI", "BLK", "AXP", "SBUX"]            # Financial services + Coffee (Day 14)
}

# Determine which day of 2-week cycle (Day 1-14)
day_index = (datetime.now() - datetime(2025, 11, 7)).days % 14  # Starting from Nov 7
day_names = ["thursday", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday",
             "thursday2", "friday2", "saturday2", "sunday2", "monday2", "tuesday2", "wednesday2"]
day_key = day_names[day_index]
SP500_STOCKS = STOCK_GROUPS[day_key]

EPOCHS = 100
CHECKPOINT_EVERY = 20

print(f"Day {day_index + 1}/14: {', '.join(SP500_STOCKS)}")

all_X = []
all_y = []
successful_stocks = []
failed_stocks = []

for i, ticker in enumerate(SP500_STOCKS, 1):
    try:
        print(f"  [{i}/{len(SP500_STOCKS)}] {ticker}...", end=" ", flush=True)
        
        X, y = gather_data(ticker, days_back=TRAINING_DATA_DAYS)
        
        if len(X) < 5:
            print(f"skipped")
            failed_stocks.append(ticker)
            continue
        
        all_X.append(X)
        all_y.append(y)
        successful_stocks.append(ticker)
        print(f"✓ {len(X)} samples")
        
    except Exception as e:
        print(f"failed")
        failed_stocks.append(ticker)

if not all_X:
    print("No data collected!")
    exit(1)

# Combine all data
X_combined = np.vstack(all_X)
y_combined = np.concatenate(all_y)

print(f"\nTraining: {len(successful_stocks)} stocks, {len(X_combined)} samples, {X_combined.shape[1]} features")

# Prepare directories
models_dir = Path("data/processed/models")
checkpoints_dir = Path("data/checkpoints")
logs_dir = Path("data/processed/training_logs")
scalers_dir = Path("data/processed/scalers")
for d in [models_dir, checkpoints_dir, logs_dir, scalers_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load existing model if available (incremental training across days)
existing_models = sorted(models_dir.glob("sp500_model_*.pth"))
X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)
input_dim = X_scaled.shape[1]

if existing_models:
    print(f"Loading model: {existing_models[-1].name}")
    try:
        model = torch.load(existing_models[-1])
    except Exception as e:
        print(f"Could not load, starting fresh")
        model = StockPredictor(input_dim)
else:
    print("Starting fresh (Day 1)")
    model = StockPredictor(input_dim)

# Train the model
print(f"Training {EPOCHS} epochs...")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)  # Proper 2D shape

training_losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    training_losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: {loss.item():.4f}")
    
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)


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
    
    print(f"\nMAE: ${mae:.2f} | MAPE: {mape:.1f}% | RMSE: ${rmse:.2f}")

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
    f.write(f"Stocks: {', '.join(successful_stocks)}\n")
    f.write(f"Samples: {len(X_combined):,}\n")
    f.write(f"Features: {X_combined.shape[1]}\n\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Final Loss: {final_loss:.6f}\n\n")
    f.write(f"MAE: ${mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write(f"RMSE: ${rmse:.2f}\n\n")
    f.write(f"Model: {model_path.name}\n")
    if failed_stocks:
        f.write(f"\nFailed: {', '.join(failed_stocks)}\n")

print(f"Done! Day {day_index + 1}/14 complete")

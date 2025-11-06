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

print("=" * 60)
print("📊 OPTIMIZED DAILY TRAINING")
print("=" * 60)
print("\n💡 This script is optimized for NewsAPI free tier:")
print("  • Trains on 3 stocks to minimize API calls")
print("  • Uses cache from previous runs")
print("  • Only fetches today's news (1-2 new requests per stock)")
print("  • Best for daily incremental updates\n")

# Minimal stock list for daily updates (saves API requests)
TICKERS = ["AAPL", "TSLA", "GOOGL"]  # 3 major stocks
EPOCHS = 50  # Faster training for daily updates

print(f"🎯 Training on: {', '.join(TICKERS)}")
print(f"⏱️  Epochs: {EPOCHS}\n")

# Gather data
print("📊 Gathering data (using cache when available)...")
X_all, y_all = [], []

for ticker in TICKERS:
    try:
        print(f"  {ticker}...", end=" ")
        X, y = gather_data(ticker, days_back=TRAINING_DATA_DAYS)
        X_all.append(X)
        y_all.append(y)
        print(f"✓ {len(X)} samples")
    except Exception as e:
        print(f"✗ {e}")

if not X_all:
    print("❌ No data collected! Check your internet connection or API key.")
    exit(1)

# Combine data
X_combined = np.vstack(X_all)
y_combined = np.concatenate(y_all)

print(f"\n📈 Dataset: {len(X_combined):,} samples from {len(X_all)} stocks")

# Prepare directories
models_dir = Path("data/processed/models")
scalers_dir = Path("data/processed/scalers")
logs_dir = Path("data/processed/training_logs")
models_dir.mkdir(parents=True, exist_ok=True)
scalers_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

# Load existing model if available for incremental training
existing_models = sorted(models_dir.glob("multi_stock_model_*.pth"))
if existing_models:
    print(f"\n📦 Found existing model: {existing_models[-1].name}")
    print("   Loading for incremental training...")
    try:
        model = torch.load(existing_models[-1])
        print("   ✓ Model loaded successfully")
    except:
        print("   ⚠️  Could not load existing model, training from scratch")
        X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)
        input_dim = X_scaled.shape[1]
        model = StockPredictor(input_dim)
else:
    print("\n🆕 No existing model found, training from scratch")
    X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)
    input_dim = X_scaled.shape[1]
    model = StockPredictor(input_dim)

# Scale data
X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)

# Training
print(f"\n🔥 Training for {EPOCHS} epochs...")
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    final_loss = criterion(predictions, y_tensor).item()
    
    predictions_actual = scaler_y.inverse_transform(predictions.numpy())
    y_actual = scaler_y.inverse_transform(y_tensor.numpy())
    
    mae = np.abs(predictions_actual - y_actual).mean()
    mape = np.abs((predictions_actual - y_actual) / y_actual).mean() * 100

print(f"\n✅ Training Complete!")
print(f"  • MAE: ${mae:.2f}")
print(f"  • MAPE: {mape:.2f}%")

# Save model with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = models_dir / f"multi_stock_model_{timestamp}.pth"
torch.save(model, model_path)
print(f"\n💾 Model saved: {model_path}")

# Save scalers
with open(scalers_dir / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(scalers_dir / "scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# Save log
log_path = logs_dir / f"training_{timestamp}.txt"
with open(log_path, "w") as f:
    f.write(f"Daily Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Stocks: {', '.join(TICKERS[:len(X_all)])}\n")
    f.write(f"Samples: {len(X_combined):,}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"MAE: ${mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")

print(f"📝 Log saved: {log_path}")
print("\n🎉 Done! Run 'python scripts/run_inference.py' to make predictions.")

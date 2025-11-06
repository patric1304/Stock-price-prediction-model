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

# S&P 500 major stocks (optimized for API limits)
SP500_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "KO", "PEP", "AVGO", "COST", "WMT", "MCD", "CSCO", "ACN",
    "TMO", "NFLX", "ABT", "CRM", "ORCL", "NKE", "INTC", "VZ", "CMCSA"
]  # 36 stocks

EPOCHS = 100
CHECKPOINT_EVERY = 20

print("=" * 70)
print("🎯 BACKTEST TRAINING: S&P 500 HISTORICAL DATA")
print("=" * 70)
print(f"\n📊 Training Strategy:")
print(f"  • Fetch last 30 days of historical data")
print(f"  • For each day: predict NEXT day's price")
print(f"  • Compare prediction with ACTUAL price (known from history)")
print(f"  • Train model by learning from prediction errors")
print(f"  • Repeat across {len(SP500_STOCKS)} S&P 500 stocks")
print(f"\n⚠️  NewsAPI Limitations:")
print(f"  • Free tier: {MAX_DAILY_REQUESTS} requests/day")
print(f"  • This training: ~{len(SP500_STOCKS) * 2} requests (first run)")
print(f"  • Subsequent runs: ~5-10 requests (cache is used)")
print("=" * 70)

# Gather historical data
print(f"\n📈 Gathering historical data from {len(SP500_STOCKS)} stocks...")
print("   Each stock provides training samples where we KNOW the outcome\n")

all_X = []
all_y = []
successful_stocks = []
failed_stocks = []

for i, ticker in enumerate(SP500_STOCKS, 1):
    try:
        print(f"  [{i:2d}/{len(SP500_STOCKS)}] {ticker:6s}...", end=" ", flush=True)
        
        X, y = gather_data(ticker, days_back=TRAINING_DATA_DAYS)
        
        if len(X) < 5:
            print(f"⚠️  Skipped (only {len(X)} samples)")
            failed_stocks.append(ticker)
            continue
        
        all_X.append(X)
        all_y.append(y)
        successful_stocks.append(ticker)
        print(f"✓ {len(X):3d} training samples")
        
    except Exception as e:
        error_msg = str(e)[:40]
        print(f"✗ Failed: {error_msg}")
        failed_stocks.append(ticker)

if not all_X:
    print("\n❌ No data collected! Check your NewsAPI key.")
    exit(1)

# Combine all data
X_combined = np.vstack(all_X)
y_combined = np.concatenate(all_y)

print(f"\n" + "=" * 70)
print(f"📊 TRAINING DATASET SUMMARY")
print(f"=" * 70)
print(f"  ✅ Successful stocks: {len(successful_stocks)}/{len(SP500_STOCKS)}")
print(f"  📈 Total training samples: {len(X_combined):,}")
print(f"  🎯 Features per sample: {X_combined.shape[1]}")
print(f"  💡 Each sample = one day predicting next day")
if failed_stocks:
    print(f"  ⚠️  Failed: {', '.join(failed_stocks[:5])}")
print("=" * 70)

# Prepare directories
models_dir = Path("data/processed/models")
checkpoints_dir = Path("data/checkpoints")
logs_dir = Path("data/processed/training_logs")
scalers_dir = Path("data/processed/scalers")
for d in [models_dir, checkpoints_dir, logs_dir, scalers_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Train the model
print(f"\n🔥 Training model for {EPOCHS} epochs...")
print(f"   Learning to predict tomorrow's price from today's data")
print("=" * 70)

X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)

input_dim = X_scaled.shape[1]
model = StockPredictor(input_dim)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

training_losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    training_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}], Loss: {loss.item():.6f}")
    
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: epoch_{epoch+1}.pth")

print("=" * 70)

# Calculate final accuracy
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    final_loss = criterion(predictions, y_tensor).item()
    
    predictions_actual = scaler_y.inverse_transform(predictions.numpy())
    y_actual = scaler_y.inverse_transform(y_tensor.numpy())
    
    mae = np.abs(predictions_actual - y_actual).mean()
    mape = np.abs((predictions_actual - y_actual) / y_actual).mean() * 100
    rmse = np.sqrt(np.mean((predictions_actual - y_actual) ** 2))
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"=" * 70)
    print(f"  📊 Performance Metrics:")
    print(f"     • Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"     • Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"     • Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"\n  💡 Interpretation:")
    print(f"     On average, predictions are off by ${mae:.2f}")
    print(f"     That's {mape:.2f}% error rate")
    print("=" * 70)

# Save final model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = models_dir / f"sp500_model_{timestamp}.pth"
torch.save(model, model_path)
print(f"\n💾 Model saved: {model_path.name}")

# Save scalers
with open(scalers_dir / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(scalers_dir / "scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)
print(f"💾 Scalers saved")

# Save training log
log_path = logs_dir / f"training_{timestamp}.txt"
with open(log_path, "w") as f:
    f.write("S&P 500 Backtest Training Log\n")
    f.write("=" * 70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Training Strategy:\n")
    f.write(f"  - Used historical data where outcomes are known\n")
    f.write(f"  - Trained to predict next-day prices\n")
    f.write(f"  - Learned from {len(successful_stocks)} S&P 500 stocks\n\n")
    f.write(f"Dataset:\n")
    f.write(f"  - Stocks: {', '.join(successful_stocks[:10])}\n")
    if len(successful_stocks) > 10:
        f.write(f"           ... and {len(successful_stocks)-10} more\n")
    f.write(f"  - Total samples: {len(X_combined):,}\n")
    f.write(f"  - Features: {X_combined.shape[1]}\n\n")
    f.write(f"Training:\n")
    f.write(f"  - Epochs: {EPOCHS}\n")
    f.write(f"  - Final Loss: {final_loss:.6f}\n\n")
    f.write(f"Performance:\n")
    f.write(f"  - MAE: ${mae:.2f}\n")
    f.write(f"  - MAPE: {mape:.2f}%\n")
    f.write(f"  - RMSE: ${rmse:.2f}\n\n")
    f.write(f"Model: {model_path.name}\n")
    
    if failed_stocks:
        f.write(f"\nFailed stocks: {', '.join(failed_stocks)}\n")

print(f"📝 Training log saved: {log_path.name}")
print("\n" + "=" * 70)
print("🎉 MODEL IS READY!")
print("=" * 70)
print("\nNext steps:")
print("  1. Check accuracy: python scripts/evaluate_model.py")
print("  2. Make prediction: python scripts/run_inference.py")
print("\n💡 The model learned from historical data and is ready to")
print("   predict tomorrow's prices for any stock!")
print("=" * 70)

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

print("=" * 70)
print("🎯 2-WEEK S&P 500 TRAINING PLAN")
print("=" * 70)
print(f"\n📅 Training Day: {day_index + 1}/14")
print(f"🎯 Today's stocks: {', '.join(SP500_STOCKS)}")
print(f"\n📊 Optimized Strategy:")
print(f"  • 20 days of company news per stock")
print(f"  • 4 stocks/day × 20 requests = 80 API calls (under 100 limit)")
print(f"  • No global sentiment (VIX captures market mood)")
print(f"  • After 14 days: 56 S&P 500 stocks trained!")
print(f"  • Model learns from MASSIVE diversity across all sectors")
print(f"  • After 7 days: Model trained on 350 different stocks!")
print(f"  • Week 2+: Switch to train_daily.py for maintenance")
print(f"\n⚠️  NewsAPI Usage:")
print(f"  • Free tier: {MAX_DAILY_REQUESTS} requests/day")
print(f"  • Today's training: ~{len(SP500_STOCKS) * 2} requests (MAXIMIZED)")
print(f"  • Using nearly 100% of daily allowance for best results!")
print("=" * 70)

# Gather historical data
print(f"\n📈 Gathering data from today's {len(SP500_STOCKS)} stocks...")
print("   Building on knowledge from previous days (if any)\n")

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

# Load existing model if available (incremental training across days)
existing_models = sorted(models_dir.glob("sp500_model_*.pth"))
X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X_combined, y_combined)
input_dim = X_scaled.shape[1]

if existing_models:
    print(f"\n📦 Found existing model: {existing_models[-1].name}")
    print("   Loading for incremental training...")
    try:
        model = torch.load(existing_models[-1])
        print("   ✓ Model loaded - will add today's stocks to knowledge base")
    except Exception as e:
        print(f"   ⚠️  Could not load model: {e}")
        print("   Starting fresh training...")
        model = StockPredictor(input_dim)
else:
    print("\n🆕 No existing model found - starting fresh (Day 1)")
    model = StockPredictor(input_dim)

# Train the model
print(f"\n🔥 Training model for {EPOCHS} epochs...")
print(f"   Learning patterns from {day_key.capitalize()}'s stocks")  # CHANGE day_name to day_key
print("=" * 70)

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
    f.write(f"Maximized Weekly Rotation Training - {day_key.capitalize()}\n")  # CHANGE day_name to day_key
    f.write("=" * 70 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Day of Week: {day_key.capitalize()}\n\n")  # CHANGE day_name to day_key
    f.write(f"Training Strategy:\n")
    f.write(f"  - Week 1: Train ~50 stocks each day (350 total)\n")
    f.write(f"  - Model learns incrementally from previous days\n")
    f.write(f"  - Maximum API usage: ~100 requests/day\n")
    f.write(f"  - After 7 days: 350 different stocks trained!\n\n")
    f.write(f"Today's Dataset:\n")
    f.write(f"  - Stocks trained: {len(successful_stocks)}\n")
    f.write(f"  - Sample: {', '.join(successful_stocks[:10])}\n")
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
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 Progress:")
print(f"  • Today: Trained on {len(successful_stocks)} stocks ({day_key.capitalize()})")  # CHANGE day_name to day_key
print(f"  • Strategy: Run this script daily for 7 days")
print(f"  • After Week 1: 350 stocks trained - MAXIMUM DIVERSITY!")
print(f"  • Week 2+: Switch to 'python scripts/train_daily.py'")
print(f"\n🎯 Next Actions:")
print(f"  • Make prediction: python scripts/run_inference.py")
print(f"  • Test accuracy: python scripts/evaluate_model.py")
print(f"  • Tomorrow: python scripts/train_continuous.py (50 more stocks!)")
print("=" * 70)


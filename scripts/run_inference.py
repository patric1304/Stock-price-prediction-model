import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data_gathering import gather_data
import numpy as np
import pickle

print("=" * 70)
print("🔮 STOCK PRICE PREDICTION")
print("=" * 70)

# Load the latest model
models_dir = Path("data/processed/models")
model_files = sorted(models_dir.glob("sp500_model_*.pth"))

if not model_files:
    print("\n❌ No trained model found!")
    print("   Train a model first using:")
    print("   python scripts/train_continuous.py")
    exit(1)

latest_model = model_files[-1]
print(f"\n📦 Using model: {latest_model.name}")

# Load scalers
scalers_dir = Path("data/processed/scalers")
try:
    with open(scalers_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(scalers_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
except FileNotFoundError:
    print("\n❌ Scalers not found!")
    print("   Train a model first using:")
    print("   python scripts/train_continuous.py")
    exit(1)

# Load model
model = torch.load(latest_model)
model.eval()

# Get ticker from user
print("\n" + "=" * 70)
ticker = input("Enter stock ticker to predict (e.g., AAPL, TSLA): ").strip().upper()
print("=" * 70)

# Gather recent data
print(f"\n📊 Fetching recent data for {ticker}...")
try:
    X, y = gather_data(ticker, days_back=60)
except Exception as e:
    print(f"\n❌ Error fetching data: {e}")
    print("\nTroubleshooting:")
    print("  • Check if ticker symbol is correct")
    print("  • Verify internet connection")
    print("  • Check if NewsAPI key is valid")
    exit(1)

if len(X) == 0:
    print(f"\n❌ No data available for {ticker}")
    exit(1)

# Use the MOST RECENT data point to predict tomorrow
X_scaled = scaler_X.transform(X[-1:, :])  # Last day's data
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict tomorrow's price
print(f"🔮 Predicting tomorrow's price...")
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

# Display results
current_price = y[-1]
change = y_pred - current_price
change_pct = (change / current_price) * 100

print("\n" + "=" * 70)
print(f"📈 PREDICTION RESULTS FOR {ticker}")
print("=" * 70)
print(f"  📅 Today's closing price:        ${current_price:.2f}")
print(f"  🔮 Predicted tomorrow's price:   ${y_pred:.2f}")
print(f"  📊 Expected change:              ${change:+.2f} ({change_pct:+.2f}%)")
print("=" * 70)

if change > 0:
    print(f"\n� Prediction: BULLISH")
    print(f"   Price expected to RISE by ${change:.2f}")
else:
    print(f"\n📉 Prediction: BEARISH")
    print(f"   Price expected to FALL by ${abs(change):.2f}")

print("\n" + "=" * 70)
print("⚠️  DISCLAIMER")
print("=" * 70)
print("This prediction is based on historical patterns and should NOT")
print("be used as the sole basis for investment decisions.")
print("Always do your own research and consult financial advisors.")
print("=" * 70)


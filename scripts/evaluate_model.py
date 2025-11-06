import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_gathering import gather_data
import torch
import numpy as np
import pickle

print("=" * 70)
print("📊 MODEL ACCURACY EVALUATION")
print("=" * 70)

# Load the latest model
models_dir = Path("data/processed/models")
model_files = sorted(models_dir.glob("sp500_model_*.pth"))
if not model_files:
    print("\n❌ No trained model found! Train a model first:")
    print("   python scripts/train_continuous.py")
    exit(1)

latest_model = model_files[-1]
print(f"\n📦 Loading model: {latest_model.name}")
model = torch.load(latest_model)
model.eval()

# Load scalers
scalers_dir = Path("data/processed/scalers")
try:
    with open(scalers_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(scalers_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
except FileNotFoundError:
    print("\n❌ Scalers not found! Train a model first:")
    print("   python scripts/train_continuous.py")
    exit(1)

# Test on a specific stock
print("\n" + "=" * 70)
ticker = input("Enter stock ticker to evaluate (e.g., AAPL): ").strip().upper()
print("=" * 70)

print(f"\n📊 Fetching historical data for {ticker}...")
try:
    X, y = gather_data(ticker, days_back=60)
except Exception as e:
    print(f"\n❌ Error fetching data: {e}")
    exit(1)

if len(X) == 0:
    print(f"\n❌ No data available for {ticker}")
    exit(1)

print(f"✓ Retrieved {len(y)} data points")

# Scale data
X_scaled = scaler_X.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Make predictions
print(f"🔮 Making predictions on historical data...")
with torch.no_grad():
    predictions_scaled = model(X_tensor).numpy()
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

# Calculate accuracy metrics
mae = np.abs(predictions - y).mean()
mape = np.abs((predictions - y) / y).mean() * 100
rmse = np.sqrt(np.mean((predictions - y) ** 2))

# Direction accuracy
if len(predictions) > 1:
    pred_direction = np.sign(predictions[1:] - predictions[:-1])
    actual_direction = np.sign(y[1:] - y[:-1])
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
else:
    direction_accuracy = 0.0

print("\n" + "=" * 70)
print(f"📈 EVALUATION RESULTS FOR {ticker}")
print("=" * 70)
print(f"  📊 Samples tested: {len(y)}")
print(f"  💵 Mean Absolute Error (MAE): ${mae:.2f}")
print(f"  📉 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  📊 Root Mean Squared Error (RMSE): ${rmse:.2f}")
if len(predictions) > 1:
    print(f"  🎯 Direction Accuracy: {direction_accuracy:.1f}%")
print("=" * 70)

print(f"\n💡 Interpretation:")
print(f"   On average, predictions for {ticker} are off by ${mae:.2f}")
print(f"   That's a {mape:.2f}% error rate")
if direction_accuracy > 0:
    print(f"   Model correctly predicted price direction {direction_accuracy:.1f}% of the time")

# Show sample predictions
print(f"\n🔍 Sample Predictions (last 5 days):")
print("=" * 70)
print(f"{'Day':<8} {'Actual Price':>12} {'Predicted':>12} {'Error':>10} {'%Error':>8}")
print("-" * 70)
for i in range(max(0, len(y)-5), len(y)):
    error = predictions[i] - y[i]
    pct_error = (error / y[i]) * 100
    print(f"Day {i+1:<3} ${y[i]:>11.2f} ${predictions[i]:>11.2f} ${error:>9.2f} {pct_error:>7.1f}%")
print("=" * 70)

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data_gathering import gather_data
from src.preprocessing import scale_features
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

# Load model
model = torch.load(latest_model, weights_only=False)
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

# Scale features FOR THIS SPECIFIC STOCK (important!)
X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X, y)

# Use the MOST RECENT data point to predict tomorrow
X_tensor = torch.tensor(X_scaled[-1:, :], dtype=torch.float32)

# Predict tomorrow's price
print(f"🔮 Predicting tomorrow's price...")
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

# Display results
current_price = float(y[-1].item() if hasattr(y[-1], 'item') else y[-1])
change = y_pred - current_price
change_pct = (change / current_price) * 100

# Analyze recent trends for reasoning
last_5_prices = [float(p.item() if hasattr(p, 'item') else p) for p in y[-5:]]
price_trend = "rising" if last_5_prices[-1] > last_5_prices[0] else "falling"
avg_recent_change = np.mean(np.diff(last_5_prices))
volatility = np.std(last_5_prices)

# Extract news sentiment from the most recent data point
# Features are: [price_history(25), sentiment_comp(1), sentiment_global(1), macro(3), vix(1)]
last_features = X[-1]
news_sentiment = float(last_features[25]) if len(last_features) > 25 else 0.0  # sentiment_comp is at index 25

print("\n" + "=" * 70)
print(f"📈 PREDICTION RESULTS FOR {ticker}")
print("=" * 70)
print(f"  📅 Today's closing price:        ${current_price:.2f}")
print(f"  🔮 Predicted tomorrow's price:   ${y_pred:.2f}")
print(f"  📊 Expected change:              ${change:+.2f} ({change_pct:+.2f}%)")
print(f"  ⚠️  Model uncertainty (MAE):     ±$15.69")
print("=" * 70)

if change > 0:
    print(f"\n📈 Prediction: BULLISH")
    print(f"   Price expected to RISE by ${change:.2f} ({change_pct:+.1f}%)")
elif change < 0:
    print(f"\n📉 Prediction: BEARISH")
    print(f"   Price expected to FALL by ${abs(change):.2f} ({abs(change_pct):.1f}%)")
else:
    print(f"\n➡️  Prediction: NEUTRAL")
    print(f"   Price expected to remain stable")

# Provide reasoning
print("\n" + "=" * 70)
print("🤔 REASONING BEHIND PREDICTION")
print("=" * 70)

# Technical indicators
print("\n📊 Technical Indicators:")
print(f"  • Recent trend: Price has been {price_trend} (${last_5_prices[0]:.2f} → ${last_5_prices[-1]:.2f})")
print(f"  • Average daily change: ${avg_recent_change:+.2f}")
print(f"  • Price volatility: ${volatility:.2f} (lower = more stable)")

# News sentiment
print("\n📰 News Sentiment Analysis:")
if news_sentiment > 0.1:
    print(f"  • Company news: POSITIVE (score: {news_sentiment:.2f})")
    print(f"    Recent headlines suggest optimistic sentiment")
elif news_sentiment < -0.1:
    print(f"  • Company news: NEGATIVE (score: {news_sentiment:.2f})")
    print(f"    Recent headlines suggest concerning sentiment")
else:
    print(f"  • Company news: NEUTRAL (score: {news_sentiment:.2f})")
    print(f"    Recent headlines are mixed or neutral")

# Model's key factors
print("\n🧠 Model's Analysis:")
if abs(change_pct) < 1:
    print(f"  • Small predicted change suggests consolidation/sideways movement")
elif abs(change_pct) > 3:
    print(f"  • Large predicted change suggests strong directional momentum")
else:
    print(f"  • Moderate predicted change aligns with typical daily movement")

if price_trend == "rising" and change > 0:
    print(f"  • Prediction continues the upward trend")
elif price_trend == "falling" and change < 0:
    print(f"  • Prediction continues the downward trend")
else:
    print(f"  • Prediction suggests a potential trend reversal")

# Show last 5 days for context
print("\n" + "=" * 70)
print("📊 Recent Price History (Last 5 days)")
print("=" * 70)
for i in range(max(0, len(y)-5), len(y)):
    day_num = i - len(y) + 6
    price_val = float(y[i].item() if hasattr(y[i], 'item') else y[i])
    print(f"  Day {day_num}: ${price_val:.2f}")

print("\n" + "=" * 70)
print("⚠️  DISCLAIMER")
print("=" * 70)
print("This prediction is based on historical patterns and should NOT")
print("be used as the sole basis for investment decisions.")
print(f"Model trained on limited data (Day 1-4/14).")
print("Accuracy will improve as more stocks are added over 14 days.")
print("Always do your own research and consult financial advisors.")
print("=" * 70)


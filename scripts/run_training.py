import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_gathering import gather_data
from src.train import train_model
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ask user for stock ticker
ticker = input("Enter stock ticker to train on (e.g., AAPL): ").strip().upper()

# Gather data (stock + VIX + news + macro)
print(f"Fetching data for {ticker}...")
X, y = gather_data(ticker)

# Train the model
print("Training the model...")
model = train_model(X, y, epochs=20)

# Ensure processed folder exists
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Save trained model
model_path = processed_dir / "stock_model.pth"
torch.save(model, model_path)
print(f"✅ Training finished. Model saved to {model_path}")

# Save training summary as text
summary_path = processed_dir / "training_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"Stock Ticker: {ticker}\n")
    f.write(f"Training samples: {len(X)}\n")
    f.write(f"Input features: {X.shape[1]}\n")
    f.write(f"Model saved to: {model_path}\n")
print(f"✅ Training summary saved to {summary_path}")

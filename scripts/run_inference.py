import torch
from src.data_gathering import gather_data
from src.model import StockPredictor

def run_inference(model, ticker):
    X, _ = gather_data(ticker)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X[-1:], dtype=torch.float32)  # predict next day based on latest window
        y_pred = model(X_tensor).item()
    print(f"Predicted next day Close for {ticker}: {y_pred:.2f}")
    return y_pred

if __name__ == "__main__":
    ticker = input("Enter stock ticker for inference: ").strip().upper()
    # Load trained model
    model_path = "data/processed/stock_model.pth"
    model = torch.load(model_path)
    run_inference(model, ticker)

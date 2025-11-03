import torch
from sklearn.metrics import mean_squared_error, r2_score
from src.dataset import StockDataset
from torch.utils.data import DataLoader

def evaluate_model(model, X_test, y_test, batch_size=32):
    model.eval()
    dataset = StockDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds_list, y_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            preds_list.append(preds)
            y_list.append(y_batch)
    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(y_list).numpy()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Evaluation results - MSE: {mse:.6f}, R2: {r2:.4f}")
    return mse, r2

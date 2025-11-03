from sklearn.metrics import mean_squared_error, r2_score
import torch

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred = model(X_tensor).numpy().flatten()
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Evaluation -> MSE: {mse:.6f}, R2: {r2:.6f}")
    return mse, r2

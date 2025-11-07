import numpy as np
import torch

def calculate_metrics(predictions, actuals):
    """
    Calculate evaluation metrics for predictions
    
    Args:
        predictions: numpy array of predicted values
        actuals: numpy array of actual values
    
    Returns:
        dict: Dictionary containing various metrics
    """
    mae = np.abs(predictions - actuals).mean()
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.abs((predictions - actuals) / actuals).mean() * 100
    
    # Direction accuracy (did we predict the right direction of movement?)
    pred_direction = np.sign(predictions[1:] - predictions[:-1])
    actual_direction = np.sign(actuals[1:] - actuals[:-1])
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }

def evaluate_model(model, X, y, scaler_X, scaler_y):
    """
    Evaluate a trained model on data
    
    Args:
        model: Trained PyTorch model
        X: Input features (unscaled)
        y: Target values (unscaled)
        scaler_X: Fitted StandardScaler for X
        scaler_y: Fitted StandardScaler for y
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Scale input
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        predictions_scaled = model(X_tensor).numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, y)
    
    return metrics, predictions

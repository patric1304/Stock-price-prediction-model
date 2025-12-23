"""
Enhanced Model Evaluation Script
Provides comprehensive analysis of model performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

from src.model import AdvancedStockPredictor
from src.data_gathering import gather_data
from src.preprocessing import scale_features
from src.dataset import StockDataset
from torch.utils.data import DataLoader


def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, input_dim, device='cpu'):
    """Load trained model and scalers"""
    import pickle
    
    # Load model
    model = AdvancedStockPredictor(input_dim=input_dim)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load scalers
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    
    return model, scaler_X, scaler_y


def evaluate_model_comprehensive(model, X, y, scaler_X, scaler_y, device='cpu', batch_size=64):
    """
    Comprehensive model evaluation with multiple metrics
    """
    # Scale data
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1))
    
    # Create dataset and loader
    dataset = StockDataset(X_scaled, y_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get predictions
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_predictions.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Inverse transform to original scale
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, predictions_original)
    mape = np.mean(np.abs((targets_original - predictions_original) / targets_original)) * 100
    
    # Directional accuracy (did we predict the right direction?)
    actual_direction = np.diff(targets_original) > 0
    pred_direction = np.diff(predictions_original) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'predictions': predictions_original,
        'targets': targets_original
    }
    
    return metrics


def plot_comprehensive_analysis(metrics, ticker, save_path=None):
    """Create comprehensive visualization of model performance"""
    
    predictions = metrics['predictions']
    targets = metrics['targets']
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Time series comparison
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(targets, label='Actual', linewidth=2, alpha=0.8)
    plt.plot(predictions, label='Predicted', linewidth=2, alpha=0.8)
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Price ($)', fontsize=11)
    plt.title(f'{ticker} - Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot with perfect prediction line
    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(targets, predictions, alpha=0.5, s=30)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)', fontsize=11)
    plt.ylabel('Predicted Price ($)', fontsize=11)
    plt.title('Prediction Accuracy', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = plt.subplot(3, 3, 3)
    residuals = predictions - targets
    plt.scatter(predictions, residuals, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Price ($)', fontsize=11)
    plt.ylabel('Residuals ($)', fontsize=11)
    plt.title('Residual Analysis', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = plt.subplot(3, 3, 4)
    errors = targets - predictions
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error ($)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Error Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Percentage error
    ax5 = plt.subplot(3, 3, 5)
    percentage_errors = (errors / targets) * 100
    plt.hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Percentage Error (%)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Q-Q plot for residual normality
    ax6 = plt.subplot(3, 3, 6)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Residual Normality)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. Prediction error over time
    ax7 = plt.subplot(3, 3, 7)
    abs_errors = np.abs(errors)
    plt.plot(abs_errors, linewidth=1.5, color='red', alpha=0.7)
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Absolute Error ($)', fontsize=11)
    plt.title('Prediction Error Over Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 8. Directional accuracy
    ax8 = plt.subplot(3, 3, 8)
    actual_changes = np.diff(targets)
    pred_changes = np.diff(predictions)
    correct_direction = (np.sign(actual_changes) == np.sign(pred_changes))
    
    window = 20
    rolling_accuracy = np.convolve(correct_direction.astype(float), 
                                   np.ones(window)/window, mode='valid') * 100
    plt.plot(rolling_accuracy, linewidth=2)
    plt.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Random (50%)')
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Directional Accuracy (%)', fontsize=11)
    plt.title(f'Rolling Directional Accuracy (window={window})', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 9. Metrics summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    metrics_text = f"""
    Performance Metrics
    {'='*30}
    
    MSE:   ${metrics['mse']:.4f}
    MAE:   ${metrics['mae']:.4f}
    RMSE:  ${metrics['rmse']:.4f}
    R²:    {metrics['r2']:.4f}
    MAPE:  {metrics['mape']:.2f}%
    
    Directional Accuracy: {metrics['directional_accuracy']:.2f}%
    
    Min Error: ${errors.min():.2f}
    Max Error: ${errors.max():.2f}
    Mean Error: ${errors.mean():.2f}
    Std Error: ${errors.std():.2f}
    """
    ax9.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Comprehensive Model Evaluation - {ticker}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained stock prediction model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--model', type=str, default='data/checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--days', type=int, default=200, help='Days of historical data')
    parser.add_argument('--output', type=str, default='data/evaluation_results.png',
                       help='Path to save evaluation plots')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Gather data
    print(f"Gathering data for {args.ticker}...")
    X, y = gather_data(args.ticker, days_back=args.days)
    print(f"Data shape: X={X.shape}, y={y.shape}\n")
    
    # For evaluation, we'll use the test split (last 15%)
    test_size = int(0.15 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"Evaluating on test set: {len(X_test)} samples\n")
    
    # Note: You'll need to save scalers during training to load them here
    # For now, we'll create them from all data (not ideal - should use training scalers)
    X_scaled, y_scaled, scaler_X, scaler_y = scale_features(X, y)
    
    # Load model
    print("Loading model...")
    model = AdvancedStockPredictor(input_dim=X.shape[1])
    
    if Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        print(f"Model loaded from {args.model}\n")
    else:
        print(f"Warning: Model file not found at {args.model}")
        print("Evaluating with random weights (for demonstration)\n")
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model_comprehensive(
        model, X_test, y_test, scaler_X, scaler_y, device
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"MSE:  ${metrics['mse']:.4f}")
    print(f"MAE:  ${metrics['mae']:.4f}")
    print(f"RMSE: ${metrics['rmse']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print("="*60 + "\n")
    
    # Plot results
    print("Generating visualizations...")
    plot_comprehensive_analysis(metrics, args.ticker, args.output)
    
    # Save metrics to JSON
    output_path = Path(args.output).parent / 'evaluation_metrics.json'
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k not in ['predictions', 'targets']}
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()

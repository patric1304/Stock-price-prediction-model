"""
Enhanced Model Evaluation Script
Provides comprehensive analysis of model performance
"""

import sys
from pathlib import Path

# Allow running this file directly via `python scripts/evaluate_advanced_model.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import re

from src.model import AdvancedStockPredictor
from src.data_gathering import gather_data
from src.dataset import StockDataset
from torch.utils.data import DataLoader


def _infer_model_kwargs_from_state_dict(state_dict: dict, input_dim: int) -> dict:
    """Best-effort reconstruction of model kwargs from a saved state_dict."""
    # Infer hidden_dim from the first linear layer in input_projection
    hidden_dim = None
    w = state_dict.get("input_projection.0.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        hidden_dim = int(w.shape[0])

    # Infer num_layers from LSTM parameter keys: lstm.weight_ih_l{k}
    layer_idxs = []
    for k in state_dict.keys():
        m = re.match(r"lstm\.weight_ih_l(\d+)$", k)
        if m:
            layer_idxs.append(int(m.group(1)))
    num_layers = (max(layer_idxs) + 1) if layer_idxs else 1

    # Infer history_days from input_dim assuming extra features are sentiment(2) + vix(1)
    extra_dim = 3
    history_days = None
    if input_dim > extra_dim and (input_dim - extra_dim) % 5 == 0:
        history_days = int((input_dim - extra_dim) // 5)

    return {
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else 256,
        "num_layers": int(num_layers),
        # Dropout has no parameters; pick a sane default for eval.
        "dropout": 0.0,
        "history_days": int(history_days) if history_days is not None else None,
    }


def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, input_dim, device='cpu'):
    """Load trained model + scalers, reconstructing architecture from checkpoint when available."""
    import pickle
    
    checkpoint = torch.load(model_path, map_location=device)

    # target_mode may live at top-level (newer checkpoints) or be absent (older checkpoints)
    target_mode = None
    if isinstance(checkpoint, dict):
        target_mode = checkpoint.get('target_mode')

    # Load model weights (support both raw state_dict and dict checkpoints)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_kwargs = checkpoint.get('model_kwargs') or {}
    else:
        state_dict = checkpoint
        model_kwargs = {}

    # Reconstruct model architecture
    if not model_kwargs:
        model_kwargs = _infer_model_kwargs_from_state_dict(state_dict, input_dim=input_dim)

    history_days = model_kwargs.get('history_days')
    if history_days is None or int(history_days) <= 0:
        model = AdvancedStockPredictor(
            input_dim=input_dim,
            hidden_dim=int(model_kwargs.get('hidden_dim', 256)),
            num_layers=int(model_kwargs.get('num_layers', 3)),
            dropout=float(model_kwargs.get('dropout', 0.3)),
        )
    else:
        model = AdvancedStockPredictor(
            input_dim=input_dim,
            hidden_dim=int(model_kwargs.get('hidden_dim', 256)),
            num_layers=int(model_kwargs.get('num_layers', 3)),
            dropout=float(model_kwargs.get('dropout', 0.3)),
            history_days=int(history_days),
        )

    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    # Load scalers
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    
    return model, scaler_X, scaler_y, target_mode


def evaluate_model_comprehensive(model, X, y, scaler_X, scaler_y, device='cpu', batch_size=64, *, current_close=None, target_mode='price'):
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
    
    # Inverse transform to original scale (these are targets in "target_mode" space)
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    mode = (target_mode or 'price').strip().lower()
    if mode not in {'price', 'delta'}:
        mode = 'price'

    # Convert into price space for metrics/plots (especially important for delta targets)
    if mode == 'delta':
        if current_close is None:
            raise ValueError("current_close is required when evaluating delta targets")
        cc = np.asarray(current_close, dtype=np.float32).reshape(-1)
        if len(cc) != len(predictions_original):
            raise ValueError(f"current_close length {len(cc)} != predictions length {len(predictions_original)}")
        predictions_price = cc + predictions_original
        targets_price = cc + targets_original
    else:
        predictions_price = predictions_original
        targets_price = targets_original
    
    # Calculate metrics
    mse = mean_squared_error(targets_price, predictions_price)
    mae = mean_absolute_error(targets_price, predictions_price)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_price, predictions_price)
    mape = np.mean(np.abs((targets_price - predictions_price) / np.maximum(np.abs(targets_price), 1e-8))) * 100
    
    # Directional accuracy (did we predict the right direction?)
    actual_direction = np.diff(targets_price) > 0
    pred_direction = np.diff(predictions_price) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'predictions': predictions_price,
        'targets': targets_price,
        'target_mode': mode,
    }
    
    return metrics


def evaluate_baseline_naive(y_true, current_close=None, **kwargs):
    """
    Naive baseline: predict next close as today's close (persistence).
    Must handle y_true being shaped (N,) or (N,1).
    """
    # ---- normalize shapes to 1D ----
    actual = np.asarray(y_true, dtype=np.float32).reshape(-1)

    if current_close is None:
        # fallback: use naive persistence on actual itself (shifted)
        pred = np.r_[actual[0], actual[:-1]]
    else:
        pred = np.asarray(current_close, dtype=np.float32).reshape(-1)

    # ---- metrics (unchanged logic, but with 1D arrays) ----
    mse = float(np.mean((pred - actual) ** 2))
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100)

    # ---- directional accuracy (diff over time axis) ----
    actual_direction = np.sign(np.diff(actual))          # (N-1,)
    pred_direction = np.sign(np.diff(pred))              # (N-1,)

    m = min(len(actual_direction), len(pred_direction))
    directional_accuracy = float(np.mean(actual_direction[:m] == pred_direction[:m]) * 100) if m > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "pred": pred,
        "actual": actual,
    }


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
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: data/checkpoints/<TICKER>/best_model.pth)')
    parser.add_argument('--scaler-x', type=str, default=None,
                       help='Path to saved feature scaler (default: data/checkpoints/<TICKER>/scaler_X.pkl)')
    parser.add_argument('--scaler-y', type=str, default=None,
                       help='Path to saved target scaler (default: data/checkpoints/<TICKER>/scaler_y.pkl)')
    parser.add_argument('--days', type=int, default=1825, help='Days of historical data')
    parser.add_argument('--output', type=str, default='data/evaluation_results.png',
                       help='Path to save evaluation plots')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Gather data
    ticker = args.ticker.upper()
    print(f"Gathering data for {ticker}...")
    X, y, meta = gather_data(ticker, days_back=args.days, return_meta=True)
    print(f"Data shape: X={X.shape}, y={y.shape}\n")
    
    # For evaluation, we'll use the test split (last 15%)
    test_size = int(0.15 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"Evaluating on test set: {len(X_test)} samples\n")

    # Resolve default artifact paths
    default_dir = Path('data/checkpoints') / ticker
    model_path = Path(args.model) if args.model else (default_dir / 'best_model.pth')
    scaler_x_path = Path(args.scaler_x) if args.scaler_x else (default_dir / 'scaler_X.pkl')
    scaler_y_path = Path(args.scaler_y) if args.scaler_y else (default_dir / 'scaler_y.pkl')

    if not model_path.exists() or not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError(
            "Missing trained artifacts. Expected files:\n"
            f"  Model:   {model_path}\n"
            f"  ScalerX: {scaler_x_path}\n"
            f"  ScalerY: {scaler_y_path}\n"
            "Run training first: python scripts/train_advanced_model.py --ticker <TICKER>"
        )

    print("Loading model and scalers...")
    model, scaler_X, scaler_y, checkpoint_target_mode = load_model_and_scalers(
        model_path=str(model_path),
        scaler_X_path=str(scaler_x_path),
        scaler_y_path=str(scaler_y_path),
        input_dim=X.shape[1],
        device=device
    )
    print(f"Loaded from {default_dir}\n")

    # If the checkpoint specifies a target mode, re-gather data to match it.
    if checkpoint_target_mode:
        desired_mode = str(checkpoint_target_mode).strip().lower()
        if desired_mode in {'price', 'delta'}:
            print(f"Checkpoint target mode: {desired_mode}")
            X, y, meta = gather_data(ticker, days_back=args.days, return_meta=True, target_mode=desired_mode)
            test_size = int(0.15 * len(X))
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            print(f"Rebuilt evaluation data for target mode '{desired_mode}'.")
        else:
            desired_mode = 'price'
    else:
        desired_mode = 'price'
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model_comprehensive(
        model, X_test, y_test, scaler_X, scaler_y, device,
        current_close=meta['current_close'][-test_size:],
        target_mode=desired_mode,
    )

    # Baseline (naive): predict next close equals current close
    baseline = evaluate_baseline_naive(
        y_true=metrics['targets'],
        current_close=meta['current_close'][-test_size:],
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
    print("\nBaseline (naive: next close = current close):")
    print(f"  RMSE: ${baseline['rmse']:.4f}")
    print(f"  MAE:  ${baseline['mae']:.4f}")
    print(f"  MAPE: {baseline['mape']:.2f}%")
    print(f"  Directional Accuracy: {baseline['directional_accuracy']:.2f}%")
    print("="*60 + "\n")
    
    # Plot results
    print("Generating visualizations...")
    plot_comprehensive_analysis(metrics, args.ticker, args.output)
    
    # Save metrics to JSON
    output_path = Path(args.output).parent / 'evaluation_metrics.json'
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k not in ['predictions', 'targets']}
    metrics_to_save['baseline_naive'] = baseline
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()

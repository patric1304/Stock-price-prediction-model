"""
Advanced Deep Learning Training Script
Train the enhanced model with proper validation and regularization
"""

import sys
from pathlib import Path

                                                                               
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import argparse
import json
from datetime import datetime

from src.data_gathering import gather_data, NewsAPIRateLimitError
from src.train import train_model_advanced
from src.model import AdvancedStockPredictor
from src.config import TARGET_MODE


NEWSAPI_RATE_LIMIT_EXIT_CODE = 42


def main() -> int:
    parser = argparse.ArgumentParser(description='Train Advanced Stock Prediction Model')
    
                    
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=1825, help='Days of historical data to gather (default: 1825 = 5 years)')
    parser.add_argument('--as-of', type=str, default=None, help='End date (YYYY-MM-DD) for data/news alignment (default: today)')
    
                                                                          
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
                        
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
                          
    parser.add_argument('--train-split', type=float, default=0.7, help='Training data proportion')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation data proportion')
    
                      
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints', 
                       help='Base directory to save checkpoints (a per-ticker subfolder will be created)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
                                          
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*70)
    print("Advanced Stock Price Prediction - Deep Learning Training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Ticker: {args.ticker}")
    print(f"  Historical Days: {args.days}")
    if args.as_of:
        print(f"  As-Of Date: {args.as_of}")
    print(f"  Target Mode: {TARGET_MODE}")
    print(f"  Model: LSTM + Attention + Residual")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  LSTM Layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Early Stopping Patience: {args.patience}")
    print(f"  Data Split: {args.train_split*100:.0f}% train / "
          f"{args.val_split*100:.0f}% val / "
          f"{(1-args.train_split-args.val_split)*100:.0f}% test")
    print()
    
                            
    use_gpu = not args.no_gpu
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
                 
    print(f"Gathering data for {args.ticker}...")
    try:
        X, y = gather_data(args.ticker, days_back=args.days, end_date=args.as_of)
        print(f"[SUCCESS] Data gathered successfully!")
        print(f"   Features: {X.shape}")
        print(f"   Targets: {y.shape}")
        print(f"   Total samples: {len(X)}")
        print()
    except NewsAPIRateLimitError as e:
        print(f"[ERROR] {e}")
        print("[ERROR] Stopping because NewsAPI free-tier rate limit was hit.")
        return NEWSAPI_RATE_LIMIT_EXIT_CODE
    except Exception as e:
        print(f"[ERROR] Error gathering data: {e}")
        return 1
    
                 
    print("Starting training...")
    print("="*70)
    
    try:
                                                                          
        ticker_checkpoint_dir = str(Path(args.checkpoint_dir) / args.ticker.upper())

        model, history, scalers, test_metrics = train_model_advanced(
            X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            train_split=args.train_split,
            val_split=args.val_split,
            patience=args.patience,
            target_mode=TARGET_MODE,
            use_gpu=use_gpu,
            checkpoint_dir=ticker_checkpoint_dir,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("[SUCCESS] Training completed successfully!")
        print("="*70)
        
                                   
        checkpoint_dir = Path(ticker_checkpoint_dir)
        
                      
        import pickle
        scaler_X, scaler_y = scalers
        with open(checkpoint_dir / 'scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(checkpoint_dir / 'scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        print(f"\n[SUCCESS] Scalers saved to {checkpoint_dir}")
        
                               
        history_path = checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"[SUCCESS] Training history saved to {history_path}")
        
                                       
        report = {
            'timestamp': datetime.now().isoformat(),
            'ticker': args.ticker,
            'configuration': {
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'epochs_max': args.epochs,
                'epochs_trained': len(history['train_loss']),
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'patience': args.patience,
                'train_split': args.train_split,
                'val_split': args.val_split,
            },
            'data': {
                'total_samples': int(len(X)),
                'feature_dim': int(X.shape[1]),
                'days_back': args.days,
            },
            'results': {
                'best_val_loss': float(min(history['val_loss'])),
                'final_train_loss': float(history['train_loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'test_metrics': test_metrics,
            }
        }
        
        report_path = checkpoint_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[SUCCESS] Training report saved to {report_path}")
        
                             
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Epochs Trained: {len(history['train_loss'])}")
        print(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        print("\nTest Set Performance:")
        mode = (TARGET_MODE or "price").strip().lower()
        if mode in {"delta", "logret"}:
            print(f"  RMSE ({mode}): {test_metrics['rmse']:.4f}")
            print(f"  MAE  ({mode}): {test_metrics['mae']:.4f}")
            print(f"  MAPE ({mode}): N/A")
        else:
            print(f"  RMSE: ${test_metrics['rmse']:.2f}")
            print(f"  MAE:  ${test_metrics['mae']:.2f}")
            print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print("="*70)
        
        print(f"\n[SUCCESS] All artifacts saved to: {checkpoint_dir}")
        print("\nTo evaluate the model, run:")
        print(f"  python scripts/evaluate_advanced_model.py --ticker {args.ticker}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

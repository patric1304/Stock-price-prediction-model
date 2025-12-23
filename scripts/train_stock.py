"""
Simple wrapper to train stocks with safety check to prevent accidental re-training.
Can train from a list file or individual ticker.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_advanced_model import main as train_main


def model_exists(ticker):
    """Check if model and scalers exist for a ticker."""
    model_path = Path(f"data/processed/models/{ticker}_model.pth")
    scaler_path = Path(f"data/processed/scalers/{ticker}_scalers.pkl")
    return model_path.exists() and scaler_path.exists()


def load_stocks_list(file_path='config/stocks_to_train.txt'):
    """Load list of stocks from file."""
    stocks = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    stocks.append(line.upper())
        return stocks
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return []


def train_stock(ticker, force=False, skip_existing=False, **kwargs):
    """
    Train a stock with safety check.
    
    Args:
        ticker: Stock ticker symbol
        force: Skip confirmation if model exists
        skip_existing: Skip without asking if model exists
        **kwargs: Additional arguments to pass to training
    """
    if model_exists(ticker):
        if skip_existing:
            print(f"[SKIPPED] Model for {ticker} already exists. Skipping...")
            return False
        elif force:
            print(f"[WARNING] Model for {ticker} exists. Force re-training...")
        else:
            response = input(f"[WARNING] Model for {ticker} already exists. Re-train? (y/n): ")
            if response.lower() != 'y':
                print("[SKIPPED] Training cancelled.")
                return False
    
    # Build arguments for training
    args = ['--ticker', ticker]
    
    if 'epochs' in kwargs and kwargs['epochs']:
        args.extend(['--epochs', str(kwargs['epochs'])])
    if 'batch_size' in kwargs and kwargs['batch_size']:
        args.extend(['--batch-size', str(kwargs['batch_size'])])
    if 'learning_rate' in kwargs and kwargs['learning_rate']:
        args.extend(['--learning-rate', str(kwargs['learning_rate'])])
    
    # Call training by modifying sys.argv
    print(f"\n[TRAINING] Starting training for {ticker}...")
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set new sys.argv for the training script
        sys.argv = ['train_advanced_model.py'] + args
        
        # Call training
        train_main()
        
        print(f"[SUCCESS] Training completed for {ticker}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Training failed for {ticker}: {str(e)}\n")
        return False
    finally:
        # Always restore original sys.argv
        sys.argv = original_argv


def train_from_list(stocks_file='config/stocks_to_train.txt', skip_existing=True, **kwargs):
    """Train all stocks from list file."""
    stocks = load_stocks_list(stocks_file)
    
    if not stocks:
        print("[ERROR] No stocks found in list.")
        return
    
    print(f"[INFO] Found {len(stocks)} stocks to train")
    print(f"[INFO] Skip existing: {skip_existing}")
    print(f"[INFO] Stocks: {', '.join(stocks)}\n")
    
    results = {'success': [], 'skipped': [], 'failed': []}
    
    for i, ticker in enumerate(stocks, 1):
        print(f"{'='*60}")
        print(f"[{i}/{len(stocks)}] Processing {ticker}")
        print(f"{'='*60}")
        
        success = train_stock(ticker, skip_existing=skip_existing, **kwargs)
        
        if success:
            results['success'].append(ticker)
        elif model_exists(ticker):
            results['skipped'].append(ticker)
        else:
            results['failed'].append(ticker)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully trained: {len(results['success'])} stocks")
    if results['success']:
        print(f"  {', '.join(results['success'])}")
    
    print(f"\nSkipped (already trained): {len(results['skipped'])} stocks")
    if results['skipped']:
        print(f"  {', '.join(results['skipped'])}")
    
    print(f"\nFailed: {len(results['failed'])} stocks")
    if results['failed']:
        print(f"  {', '.join(results['failed'])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train stock models - single ticker or from list file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single stock
  python scripts/train_stock.py AAPL
  
  # Train all stocks from list (skips already trained)
  python scripts/train_stock.py --from-list
  
  # Train all stocks, re-train existing ones
  python scripts/train_stock.py --from-list --retrain
  
  # Train from custom list file
  python scripts/train_stock.py --from-list --stocks-file my_stocks.txt
        """
    )
    
    # Main arguments
    parser.add_argument('ticker', type=str, nargs='?', 
                       help='Stock ticker symbol (e.g., AAPL) - not needed with --from-list')
    parser.add_argument('--from-list', action='store_true',
                       help='Train all stocks from list file')
    parser.add_argument('--stocks-file', type=str, default='config/stocks_to_train.txt',
                       help='Path to stocks list file (default: config/stocks_to_train.txt)')
    
    # Training options
    parser.add_argument('--epochs', type=int, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (default: 0.001)')
    
    # Safety options
    parser.add_argument('--force', action='store_true', 
                       help='Force re-training without confirmation (single ticker only)')
    parser.add_argument('--retrain', action='store_true',
                       help='Re-train existing models when using --from-list')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_list:
        # Train from list
        skip_existing = not args.retrain
        train_from_list(
            stocks_file=args.stocks_file,
            skip_existing=skip_existing,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    elif args.ticker:
        # Train single ticker
        train_stock(
            args.ticker,
            force=args.force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        parser.print_help()

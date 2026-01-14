"""Deep (few stocks, lots of history) experiment trainer.

Goal: Train a small set of tickers with ~5 years of data (more samples), while
respecting NewsAPI limitations:
- Sentiment is forced to 0 before the last N days (default 30)
- Only the most recent N days include news + sentiment analysis

This is intentionally separate from existing scripts.

Examples:
  python scripts/train_deep_experiment_model.py --from-file
  python scripts/train_deep_experiment_model.py --tickers AAPL MSFT AMZN JPM
  python scripts/train_deep_experiment_model.py --from-file --checkpoint-dir data/checkpoints_deep_5y
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HISTORY_DAYS, TARGET_MODE
from src.train import train_model_advanced
from src.data_gathering_deep_experiment import NewsAPIRateLimitError, gather_data_deep_experiment
from src.deep_experiment_dataset_cache import (
    DeepDatasetCacheKey,
    effective_as_of_label,
    load_dataset_parquet,
    make_cache_path,
    save_dataset_parquet,
)


NEWSAPI_RATE_LIMIT_EXIT_CODE = 42


def _load_tickers_file(path: str) -> list[str]:
    tickers: list[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s.upper())
    return tickers


def train_one(
    *,
    ticker: str,
    days: int,
    news_days: int,
    as_of: str | None,
    target_mode: str,
    checkpoint_root: Path,
    use_gpu: bool,
    seed: int,
    dataset_cache_dir: Path,
    use_dataset_cache: bool,
    rebuild_dataset_cache: bool,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    train_split: float,
    val_split: float,
):
    print("=" * 70)
    print(f"DEEP EXPERIMENT TRAINING | {ticker}")
    print("=" * 70)

    # Dataset caching (Parquet)
    as_of_label = effective_as_of_label(as_of)
    history_days = int(HISTORY_DAYS)

    cache_key = DeepDatasetCacheKey(
        ticker=ticker.upper(),
        as_of=as_of_label,
        days_back=int(days),
        history_days=int(history_days),
        target_mode=str(target_mode),
        news_days=int(news_days),
    )
    cache_path = make_cache_path(dataset_cache_dir, cache_key)

    meta: dict = {}
    if use_dataset_cache and cache_path.exists() and not rebuild_dataset_cache:
        print(f"Loading cached dataset: {cache_path}")
        X, y, meta = load_dataset_parquet(cache_path)
        print(f"[SUCCESS] Cached data: X={tuple(X.shape)} y={tuple(y.shape)}")
    else:
        print(f"Gathering 5y data for {ticker} (days={days})...")
        X, y, meta = gather_data_deep_experiment(
            ticker,
            days_back=days,
            news_history_days=news_days,
            strict_news_cutoff=True,
            return_meta=True,
            target_mode=target_mode,
            end_date=as_of,
        )
        print(f"[SUCCESS] Data: X={tuple(X.shape)} y={tuple(y.shape)}")

        if use_dataset_cache:
            try:
                save_dataset_parquet(
                    path=cache_path,
                    X=X,
                    y=y,
                    current_close=meta.get("current_close"),
                    target_date=meta.get("target_date"),
                )
                print(f"[SUCCESS] Saved dataset cache: {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to save Parquet cache ({cache_path}): {e}")
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    ticker_dir = checkpoint_root / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    model, history, scalers, test_metrics = train_model_advanced(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        train_split=train_split,
        val_split=val_split,
        patience=patience,
        target_mode=target_mode,
        use_gpu=use_gpu,
        checkpoint_dir=str(ticker_dir),
        verbose=True,
    )

    scaler_X, scaler_y = scalers
    with open(ticker_dir / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(ticker_dir / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    with open(ticker_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    report = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "experiment": {
            "name": "deep_experiment_5y",
            "days_back": int(days),
            "news_history_days": int(news_days),
            "strict_news_cutoff": True,
        },
        "target_mode": target_mode,
        "configuration": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": float(dropout),
            "epochs_max": epochs,
            "epochs_trained": len(history.get("train_loss", [])),
            "batch_size": batch_size,
            "learning_rate": lr,
            "patience": patience,
            "train_split": train_split,
            "val_split": val_split,
            "as_of": as_of,
            "seed": seed,
        },
        "data": {
            "total_samples": int(len(X)),
            "feature_dim": int(X.shape[1]),
        },
        "results": {
            "test_metrics": test_metrics,
        },
    }

    with open(ticker_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Saved artifacts to: {ticker_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deep experiment: train few tickers with 5y history; news only last 30 days",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--from-file", action="store_true", help="Train tickers from a file")
    parser.add_argument(
        "--tickers-file",
        type=str,
        default="config/stocks_deep_experiment.txt",
        help="File containing tickers (one per line)",
    )
    parser.add_argument("--tickers", nargs="*", default=None, help="Explicit list of tickers")

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="How many years of history to use (calendar years; converted to days)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override history length in days (if set, takes precedence over --years)",
    )
    parser.add_argument("--news-days", type=int, default=30, help="Only last N days use news sentiment")
    parser.add_argument("--as-of", type=str, default=None, help="End date (YYYY-MM-DD), default today")

    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default="data/processed/deep_experiment_datasets",
        help="Where to store Parquet datasets for deep experiment",
    )
    parser.add_argument(
        "--no-dataset-cache",
        action="store_true",
        help="Disable saving/loading Parquet dataset cache",
    )
    parser.add_argument(
        "--rebuild-dataset-cache",
        action="store_true",
        help="Rebuild dataset even if a cached Parquet exists",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/checkpoints_deep_5y",
        help="Separate checkpoint root for this experiment",
    )

    # Model/training knobs (defaults chosen to be similar to your advanced script)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)

    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default=None, help="price|delta|logret (defaults to env/config)")

    args = parser.parse_args()

    target_mode = (args.target_mode or TARGET_MODE or "logret").strip().lower()

    days_back = int(args.days) if args.days is not None else int(args.years * 365)

    tickers: list[str]
    if args.tickers and len(args.tickers) > 0:
        tickers = [t.upper() for t in args.tickers]
    elif args.from_file or True:
        # Default to file so running without args "just works".
        tickers = _load_tickers_file(args.tickers_file)
    else:
        tickers = []

    if not tickers:
        print("[ERROR] No tickers provided.")
        return 2

    use_gpu = not args.no_gpu
    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    dataset_cache_dir = Path(args.dataset_cache_dir)

    print("=" * 70)
    print("DEEP EXPERIMENT (WIDE vs DEEP) TRAINING")
    print("=" * 70)
    print(f"Tickers: {', '.join(tickers)}")
    if args.days is None:
        print(f"Years back: {args.years} (~{days_back} days)")
    else:
        print(f"Days back: {days_back} (overrides --years)")
    print(f"News days: {args.news_days} (strict cutoff)")
    print(f"Target mode: {target_mode}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"GPU: {'ON' if use_gpu and torch.cuda.is_available() else 'OFF'}")
    if args.as_of:
        print(f"As-of: {args.as_of}")

    for ticker in tickers:
        try:
            train_one(
                ticker=ticker,
                days=days_back,
                news_days=args.news_days,
                as_of=args.as_of,
                target_mode=target_mode,
                checkpoint_root=checkpoint_root,
                use_gpu=use_gpu,
                seed=args.seed,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                train_split=args.train_split,
                val_split=args.val_split,
                dataset_cache_dir=dataset_cache_dir,
                use_dataset_cache=(not args.no_dataset_cache),
                rebuild_dataset_cache=bool(args.rebuild_dataset_cache),
            )
        except NewsAPIRateLimitError as e:
            print(f"[ERROR] {e}")
            print("[ERROR] NewsAPI rate limit hit; stop and re-run later.")
            return NEWSAPI_RATE_LIMIT_EXIT_CODE
        except Exception as e:
            print(f"[ERROR] Failed for {ticker}: {e}")
            # continue to next ticker to avoid wasting the run
            continue

    print("\n[SUCCESS] Deep experiment training run finished.")
    print("To predict using these models:")
    print(f"  python scripts/predict_next_close.py --ticker AAPL --checkpoint-root {checkpoint_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

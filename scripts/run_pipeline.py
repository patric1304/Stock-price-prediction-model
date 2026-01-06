"""Batch training + evaluation runner.

Designed for coursework workflows:
- Train/evaluate a single ticker or a whole list from config/stocks_to_train.txt
- Avoid retyping long CLI commands
- Optionally disable NewsAPI usage to respect request limits

Examples:
  # Train+eval one ticker
  python scripts/run_pipeline.py --ticker AAPL

  # Train+eval tickers from list (default file)
  python scripts/run_pipeline.py --from-list

  # Train first 10 tickers from list, no NewsAPI
  python scripts/run_pipeline.py --from-list --limit 10 --disable-news
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def load_tickers_list(file_path: str) -> List[str]:
    tickers: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    return tickers


def artifacts_exist(ticker: str, checkpoint_base: Path) -> bool:
    base = checkpoint_base / ticker.upper()
    return (base / "best_model.pth").exists() and (base / "scaler_X.pkl").exists() and (base / "scaler_y.pkl").exists()


NEWSAPI_RATE_LIMIT_EXIT_CODE = 42


def run(cmd: List[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False, env=env, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate stock models (single or list)")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--ticker", type=str, help="Single ticker to train/evaluate")
    group.add_argument(
        "--from-list",
        action="store_true",
        help="Train/evaluate tickers from a list file (default: config/stocks_to_train.txt)",
    )

    parser.add_argument(
        "--tickers-file",
        type=str,
        default="config/stocks_to_train.txt",
        help="Ticker list file (used with --from-list)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N tickers from the list")

    # Training args (kept minimal; pass-through)
    parser.add_argument("--days", type=int, default=500, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    parser.add_argument(
        "--checkpoint-base",
        type=str,
        default="data/checkpoints",
        help="Base folder where per-ticker checkpoints are stored",
    )

    parser.add_argument("--skip-existing", action="store_true", help="Skip tickers that already have artifacts")
    parser.add_argument("--disable-news", action="store_true", help="Disable NewsAPI usage (sets NEWS_API_KEY empty for this run)")
    parser.add_argument("--no-eval", action="store_true", help="Train only (skip evaluation)")

    args = parser.parse_args()

    root = Path(__file__).parent.parent
    checkpoint_base = root / args.checkpoint_base

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Default to list mode if no ticker provided (coursework convenience)
        if not args.from_list:
            args.from_list = True
        list_path = root / args.tickers_file
        if not list_path.exists():
            print(f"[ERROR] Ticker list file not found: {list_path}")
            return 2
        tickers = load_tickers_list(str(list_path))

    if args.limit is not None:
        tickers = tickers[: max(0, args.limit)]

    env = os.environ.copy()
    if args.disable_news:
        env["NEWS_API_KEY"] = ""

    train_script = str(root / "scripts" / "train_advanced_model.py")
    eval_script = str(root / "scripts" / "evaluate_advanced_model.py")

    ok, skipped, failed = [], [], []

    for i, ticker in enumerate(tickers, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(tickers)}] {ticker}")
        print("=" * 70)

        if args.skip_existing and artifacts_exist(ticker, checkpoint_base=checkpoint_base):
            print(f"[SKIP] Artifacts already exist for {ticker}")
            skipped.append(ticker)
            continue

        train_proc = run(
                [
                    sys.executable,
                    train_script,
                    "--ticker",
                    ticker,
                    "--days",
                    str(args.days),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--num-layers",
                    str(args.num_layers),
                    "--dropout",
                    str(args.dropout),
                    "--lr",
                    str(args.lr),
                    "--patience",
                    str(args.patience),
                    "--checkpoint-dir",
                    str(checkpoint_base),
                ],
                env=env,
            )

        if train_proc.returncode == NEWSAPI_RATE_LIMIT_EXIT_CODE:
            print("[STOP] NewsAPI rate limit detected. Stopping batch run.")
            print("       Tip: re-run with --disable-news, or wait and try again later.")
            failed.append(ticker)
            break

        if train_proc.returncode != 0:
            failed.append(ticker)
            continue

        if not args.no_eval:
            out_png = str(root / "data" / f"{ticker}_eval.png")
            eval_proc = run(
                    [
                        sys.executable,
                        eval_script,
                        "--ticker",
                        ticker,
                        "--days",
                        str(args.days),
                        "--output",
                        out_png,
                    ],
                    env=env,
                )

            if eval_proc.returncode != 0:
                failed.append(ticker)
                continue

        ok.append(ticker)

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Success: {len(ok)}")
    if ok:
        print("  " + ", ".join(ok))
    print(f"Skipped: {len(skipped)}")
    if skipped:
        print("  " + ", ".join(skipped))
    print(f"Failed: {len(failed)}")
    if failed:
        print("  " + ", ".join(failed))

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

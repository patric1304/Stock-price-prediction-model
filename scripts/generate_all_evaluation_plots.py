"""Batch-generate evaluation plots for all trained tickers.

This script discovers per-ticker folders containing:
- best_model.pth
- scaler_X.pkl
- scaler_y.pkl

Then runs scripts/evaluate_advanced_model.py programmatically in no-show mode,
so plots/metrics are saved into each ticker folder.

Examples:
  python scripts/generate_all_evaluation_plots.py --old-root data/checkpoints_logret
  python scripts/generate_all_evaluation_plots.py --old-root data/checkpoints_logret --deep-root data/checkpoints_deep_5y
  python scripts/generate_all_evaluation_plots.py --deep-root data/checkpoints_deep_5y --only AAPL MSFT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_trained_ticker_dir(ticker_dir: Path) -> bool:
    return (
        ticker_dir.is_dir()
        and (ticker_dir / "best_model.pth").exists()
        and (ticker_dir / "scaler_X.pkl").exists()
        and (ticker_dir / "scaler_y.pkl").exists()
    )


def _discover_tickers(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    tickers = []
    for child in root.iterdir():
        if _is_trained_ticker_dir(child):
            tickers.append(child.name.upper())
    return sorted(set(tickers))


def _run_eval(
    *,
    ticker: str,
    checkpoint_root: Path,
    days: int,
    as_of: str | None,
    data_source: str,
    news_days: int,
) -> int:
    # Import main only when needed.
    from scripts import evaluate_advanced_model

    argv = [
        "evaluate_advanced_model.py",
        "--ticker",
        ticker,
        "--checkpoint-root",
        str(checkpoint_root),
        "--days",
        str(days),
        "--no-show",
        "--data-source",
        data_source,
    ]
    if as_of:
        argv += ["--as-of", as_of]
    if data_source == "deep":
        argv += ["--news-days", str(news_days)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(evaluate_advanced_model.main())
    finally:
        sys.argv = old_argv


def _batch(root: Path, *, label: str, data_source: str, days: int, as_of: str | None, news_days: int, only: set[str] | None) -> None:
    tickers = _discover_tickers(root)
    if only is not None:
        tickers = [t for t in tickers if t in only]

    if not tickers:
        print(f"[INFO] No trained tickers found under {root} for {label}.")
        return

    print("=" * 70)
    print(f"BATCH EVALUATION PLOTS | {label}")
    print("=" * 70)
    print(f"Root: {root}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Days: {days}")
    if as_of:
        print(f"As-of: {as_of}")
    print(f"Data source: {data_source}")
    if data_source == "deep":
        print(f"Deep news-days: {news_days}")

    ok = 0
    fail = 0
    for i, t in enumerate(tickers, 1):
        print("-" * 70)
        print(f"[{i}/{len(tickers)}] {t}")
        try:
            code = _run_eval(
                ticker=t,
                checkpoint_root=root,
                days=days,
                as_of=as_of,
                data_source=data_source,
                news_days=news_days,
            )
            if code == 0:
                ok += 1
            else:
                fail += 1
                print(f"[WARN] Eval returned code {code} for {t}")
        except Exception as e:
            fail += 1
            print(f"[ERROR] Failed for {t}: {e}")

    print("=" * 70)
    print(f"DONE | {label}")
    print(f"Success: {ok} | Failed: {fail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots for all trained tickers (old + deep)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--old-root", type=str, default="data/checkpoints_logret", help="Existing trained checkpoints root")
    parser.add_argument("--deep-root", type=str, default="data/checkpoints_deep_5y", help="Deep experiment checkpoints root")
    parser.add_argument("--days", type=int, default=1825, help="Days of historical data for evaluation")
    parser.add_argument("--as-of", type=str, default=None, help="End date (YYYY-MM-DD), default today")
    parser.add_argument("--deep-news-days", type=int, default=30, help="Deep evaluation: only last N days use news sentiment")
    parser.add_argument("--only", nargs="*", default=None, help="Only evaluate these tickers")
    parser.add_argument("--skip-old", action="store_true", help="Skip old-root batch")
    parser.add_argument("--skip-deep", action="store_true", help="Skip deep-root batch")

    args = parser.parse_args()

    only = set(t.upper() for t in args.only) if args.only else None

    if not args.skip_old:
        _batch(
            Path(args.old_root),
            label="PAST TRAINED (OLD)",
            data_source="standard",
            days=args.days,
            as_of=args.as_of,
            news_days=args.deep_news_days,
            only=only,
        )

    if not args.skip_deep:
        _batch(
            Path(args.deep_root),
            label="DEEP EXPERIMENT (NEW)",
            data_source="deep",
            days=args.days,
            as_of=args.as_of,
            news_days=args.deep_news_days,
            only=only,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

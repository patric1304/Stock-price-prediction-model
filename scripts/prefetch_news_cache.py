"""Prefetch NewsAPI headlines into the on-disk cache, respecting request budgets.

Why
- NewsAPI free tier is rate-limited (e.g., 50 requests / 12h).
- Training loops can touch many days/tickers; this script warms the cache first.

What it does
- For each ticker and each trading day in the selected range, it fetches headlines
  for `query_date = trading_day - 1 calendar day` (same alignment as gather_data).
- It ONLY performs a live API request when the cache file is missing.
- If `NEWS_API_KEY` is not set, it becomes a cache-audit tool (no live requests).

Usage
- One ticker, last 20 trading days worth of query-dates:
  python scripts/prefetch_news_cache.py --ticker AAPL

- A list of tickers (default file) with a strict budget:
  python scripts/prefetch_news_cache.py --from-list --max-requests 50

- Just audit (no live calls):
  set NEWS_API_KEY=   (or omit it)
  python scripts/prefetch_news_cache.py --from-list
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

                                  
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NEWS_HISTORY_DAYS
from src.data_gathering import CACHE_DIR, NewsAPIRateLimitError, fetch_newsapi_headlines


@dataclass
class PrefetchStats:
    considered: int = 0
    cache_hits: int = 0
    fetched: int = 0
    skipped_budget: int = 0
    errors: int = 0


def _load_tickers_list(file_path: str) -> list[str]:
    tickers: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    return tickers


def _iter_trading_days(index: Iterable[pd.Timestamp], start: datetime, end: datetime) -> list[pd.Timestamp]:
    return [d for d in index if start.date() <= d.date() <= end.date()]


def _cache_file_for(query: str, date_str: str) -> Path:
    return CACHE_DIR / f"{query}_{date_str}_{date_str}.json"


def prefetch_for_ticker(
    ticker: str,
    start: datetime,
    end: datetime,
    max_requests_remaining: int,
    include_global: bool,
) -> tuple[PrefetchStats, int]:
    stats = PrefetchStats()

    df = yf.download(ticker, start=start - timedelta(days=5), end=end + timedelta(days=5), progress=False, auto_adjust=False)
    if df.empty:
        stats.errors += 1
        return stats, max_requests_remaining

    for dt in _iter_trading_days(df.index, start, end):
        query_dt = dt - timedelta(days=1)
        date_str = query_dt.strftime("%Y-%m-%d")

                 
        stats.considered += 1
        cf = _cache_file_for(ticker, date_str)
        if cf.exists():
            stats.cache_hits += 1
        else:
            if not os.getenv("NEWS_API_KEY"):
                                    
                continue
            if max_requests_remaining <= 0:
                stats.skipped_budget += 1
                continue
            try:
                fetch_newsapi_headlines(ticker, date_str, date_str)
                stats.fetched += 1
                max_requests_remaining -= 1
            except NewsAPIRateLimitError:
                raise
            except Exception:
                stats.errors += 1

                                       
        if include_global:
            stats.considered += 1
            gf = _cache_file_for("global economy", date_str)
            if gf.exists():
                stats.cache_hits += 1
            else:
                if not os.getenv("NEWS_API_KEY"):
                    continue
                if max_requests_remaining <= 0:
                    stats.skipped_budget += 1
                    continue
                try:
                    fetch_newsapi_headlines("global economy", date_str, date_str)
                    stats.fetched += 1
                    max_requests_remaining -= 1
                except NewsAPIRateLimitError:
                    raise
                except Exception:
                    stats.errors += 1

    return stats, max_requests_remaining


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch NewsAPI headlines into cache")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Single ticker")
    group.add_argument("--tickers", nargs="+", help="One or more tickers")
    group.add_argument("--from-list", action="store_true", help="Use config/stocks_to_train.txt")

    parser.add_argument("--tickers-file", type=str, default="config/stocks_to_train.txt")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--days",
        type=int,
        default=NEWS_HISTORY_DAYS,
        help="How many calendar days back to cover (default: NEWS_HISTORY_DAYS)",
    )
    parser.add_argument("--as-of", type=str, default=None, help="End date (YYYY-MM-DD) for prefetch window (default: today)")
    parser.add_argument("--max-requests", type=int, default=100000, help="Budget for live NewsAPI requests")
    parser.add_argument("--include-global", action="store_true", help="Also prefetch 'global economy' query")

    args = parser.parse_args()

    root = PROJECT_ROOT
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    else:
        list_path = root / args.tickers_file
        if not list_path.exists():
            print(f"[ERROR] Ticker list file not found: {list_path}")
            return 2
        tickers = _load_tickers_list(str(list_path))

    if args.limit is not None:
        tickers = tickers[: max(0, args.limit)]

    end = datetime.today() if not args.as_of else datetime.strptime(args.as_of, "%Y-%m-%d")
    start = end - timedelta(days=int(args.days))

    print(f"NEWS_API_KEY set: {bool(os.getenv('NEWS_API_KEY'))}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Tickers: {len(tickers)}")
    print(f"Date window: {start.date()} -> {end.date()} (query uses day-1)")
    print(f"Live request budget: {args.max_requests}")

    remaining = int(args.max_requests)
    totals = PrefetchStats()

    try:
        for i, t in enumerate(tickers, 1):
            print("\n" + "-" * 70)
            print(f"[{i}/{len(tickers)}] {t}")
            s, remaining = prefetch_for_ticker(
                ticker=t,
                start=start,
                end=end,
                max_requests_remaining=remaining,
                include_global=bool(args.include_global),
            )
            print(
                f"considered={s.considered} cache_hits={s.cache_hits} fetched={s.fetched} skipped_budget={s.skipped_budget} errors={s.errors}"
            )
            totals.considered += s.considered
            totals.cache_hits += s.cache_hits
            totals.fetched += s.fetched
            totals.skipped_budget += s.skipped_budget
            totals.errors += s.errors

            if remaining <= 0 and os.getenv("NEWS_API_KEY"):
                print("[STOP] Request budget exhausted.")
                break

    except NewsAPIRateLimitError as e:
        print(f"[STOP] NewsAPI rate limit detected: {e}")
        return 42

    print("\n" + "=" * 70)
    print("PREFETCH SUMMARY")
    print("=" * 70)
    print(
        f"considered={totals.considered} cache_hits={totals.cache_hits} fetched={totals.fetched} skipped_budget={totals.skipped_budget} errors={totals.errors}"
    )
    print(f"Budget remaining: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

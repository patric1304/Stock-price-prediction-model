"""Analyze whether cached NewsAPI headlines correlate with price moves.

Goal
- Build a simple, explainable dataset: for each trading day, collect that day's headlines
  (from NewsAPI cache or live API if configured) and label the next-day move (up/down).
- Report basic statistics:
  - correlation between daily sentiment and next-day return
  - accuracy of sentiment sign vs next-day direction
  - examples of days with strongest positive/negative sentiment

This does NOT train the stock model; it's a separate diagnostic tool.

Usage (PowerShell)
- `C:/.../python.exe scripts/analyze_news_correlation.py --ticker AAPL --start 2025-12-04 --end 2025-12-26`

Notes
- Headlines are taken from `data/raw/news_cache` if present.
- If `NEWS_API_KEY` is set, missing cache entries can be fetched.
"""

from __future__ import annotations

import sys
from pathlib import Path

                                                                                   
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from src.data_gathering import fetch_newsapi_headlines, compute_sentiment_score


@dataclass(frozen=True)
class DayRow:
    date: str
    next_return: float
    next_direction: int
    n_headlines: int
    sentiment: float
    sample_headlines: str


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _safe_join(headlines: list[str], max_chars: int = 500) -> str:
    cleaned = [h.strip().replace("\n", " ") for h in headlines if isinstance(h, str) and h.strip()]
    joined = " | ".join(cleaned)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 3] + "..."


def _iter_trading_days(index: Iterable[pd.Timestamp], start: datetime, end: datetime) -> list[pd.Timestamp]:
    days = [d for d in index if start.date() <= d.date() <= end.date()]
    return days


def build_dataset(ticker: str, start: str, end: str) -> pd.DataFrame:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)

                                                              
    price_start = start_dt - timedelta(days=5)
    price_end = end_dt + timedelta(days=5)
    df = yf.download(ticker, start=price_start, end=price_end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker} in range")

    close = df["Close"].copy()
    if hasattr(close, "iloc") and isinstance(close.iloc[0], (pd.Series, pd.DataFrame)):
        close = close.iloc[:, 0]

    trading_days = _iter_trading_days(df.index, start_dt, end_dt)

    rows: list[DayRow] = []
    for dt in trading_days:
                                                 
        try:
            i = df.index.get_loc(dt)
        except KeyError:
            continue
        if isinstance(i, slice):
            i = i.start
        if i is None or i >= len(df.index) - 1:
            continue

        today_close = float(close.iloc[i])
        next_close = float(close.iloc[i + 1])
        next_ret = (next_close - today_close) / max(abs(today_close), 1e-8)
        next_dir = 1 if next_ret > 0 else 0

        date_str = dt.strftime("%Y-%m-%d")
        headlines = fetch_newsapi_headlines(ticker, date_str, date_str)
        sentiment = compute_sentiment_score(headlines)

        rows.append(
            DayRow(
                date=date_str,
                next_return=float(next_ret),
                next_direction=int(next_dir),
                n_headlines=int(len(headlines)),
                sentiment=float(sentiment),
                sample_headlines=_safe_join(headlines),
            )
        )

    out = pd.DataFrame([r.__dict__ for r in rows])
    return out


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows (no trading days or no price data)."

    mask = df["n_headlines"] > 0
    df_with = df.loc[mask].copy()

    lines: list[str] = []
    lines.append(f"Rows: {len(df)} (with headlines: {len(df_with)})")

    if len(df_with) >= 3:
        corr = float(np.corrcoef(df_with["sentiment"].to_numpy(), df_with["next_return"].to_numpy())[0, 1])
        lines.append(f"Pearson corr(sentiment, next_return): {corr:.4f}")

        sent_dir = (df_with["sentiment"] > 0).astype(int)
        acc = float((sent_dir == df_with["next_direction"]).mean())
        lines.append(f"Directional accuracy: {acc:.3f} (sentiment>0 predicts UP)")

        top_pos = df_with.sort_values(["sentiment"], ascending=False).head(3)
        top_neg = df_with.sort_values(["sentiment"], ascending=True).head(3)

        lines.append("\nTop positive-sentiment days:")
        for _, r in top_pos.iterrows():
            lines.append(
                f"- {r['date']}: sentiment={r['sentiment']:.3f}, next_ret={r['next_return']:+.3%}, n={int(r['n_headlines'])}"
            )

        lines.append("Top negative-sentiment days:")
        for _, r in top_neg.iterrows():
            lines.append(
                f"- {r['date']}: sentiment={r['sentiment']:.3f}, next_ret={r['next_return']:+.3%}, n={int(r['n_headlines'])}"
            )
    else:
        lines.append("Not enough headline-days to compute correlation (need >= 3).")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/processed/news_correlation_<TICKER>_<START>_<END>.csv)",
    )
    args = parser.parse_args()

    df = build_dataset(args.ticker, args.start, args.end)

    out_path = (
        Path(args.out)
        if args.out
        else Path("data/processed")
        / f"news_correlation_{args.ticker}_{args.start}_{args.end}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(summarize(df))
    print(f"\n[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

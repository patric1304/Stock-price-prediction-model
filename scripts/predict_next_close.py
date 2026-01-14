"""CLI: predict next trading day's close % change for a trained ticker.

This script is intentionally standalone (separate from training scripts).
It loads a per-ticker checkpoint + scalers, rebuilds the latest feature vector
via `src.data_gathering.gather_data`, and outputs the predicted % move.

Examples:
  python scripts/predict_next_close.py --list
  python scripts/predict_next_close.py --ticker AAPL
  python scripts/predict_next_close.py --ticker AAPL --days 365
  python scripts/predict_next_close.py --ticker AAPL --checkpoint-root data/checkpoints_logret
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HISTORY_DAYS, TARGET_MODE


def _business_day_after(date_str: str) -> str:
    """Best-effort next trading day label.

    We avoid calling market calendars here; this is just for display.
    """
    try:
        import pandas as pd

        dt = pd.to_datetime(date_str)
        return (dt + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    except Exception:
        return date_str


def _candidate_checkpoint_roots(user_root: Optional[str]) -> list[Path]:
    if user_root:
        return [Path(user_root)]

    # Prefer the default training location, but also support your existing folders.
    return [
        Path("data/checkpoints"),
        Path("data/checkpoints_logret"),
        Path("data/checkpoints_price_old"),
    ]


def _is_valid_ticker_dir(ticker_dir: Path) -> bool:
    return (
        (ticker_dir / "best_model.pth").exists()
        and (ticker_dir / "scaler_X.pkl").exists()
        and (ticker_dir / "scaler_y.pkl").exists()
    )


def list_trained_tickers(roots: Iterable[Path]) -> list[str]:
    tickers: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and _is_valid_ticker_dir(child):
                tickers.add(child.name.upper())
    return sorted(tickers)


def resolve_ticker_dir(ticker: str, roots: Iterable[Path]) -> Path:
    t = ticker.upper().strip()
    for root in roots:
        candidate = root / t
        if _is_valid_ticker_dir(candidate):
            return candidate
    raise FileNotFoundError(
        f"No trained artifacts found for {t}. Expected best_model.pth + scaler_X.pkl + scaler_y.pkl under one of: "
        + ", ".join(str(r) for r in roots)
    )


def load_scalers(ticker_dir: Path):
    with open(ticker_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(ticker_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    return scaler_X, scaler_y


def load_model_from_checkpoint(ticker_dir: Path, *, device: torch.device):
    # Lazy import to keep --list fast.
    from src.model import AdvancedStockPredictor, StockPredictor

    checkpoint_path = ticker_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Expected format from src/train.py
    target_mode = checkpoint.get("target_mode", None)
    model_kwargs = checkpoint.get("model_kwargs", None)
    state = checkpoint.get("model_state_dict", None)

    if state is None:
        # Allow raw state_dict (older saves)
        if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            state = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    model: torch.nn.Module
    if model_kwargs and isinstance(model_kwargs, dict) and "input_dim" in model_kwargs:
        model = AdvancedStockPredictor(**model_kwargs)
    else:
        # Fallback: infer input_dim from state_dict or scalers
        # Try to infer from first linear layer of StockPredictor.
        inferred_input_dim = None
        for k, v in state.items():
            if k.endswith("network.0.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                inferred_input_dim = int(v.shape[1])
                break
        if inferred_input_dim is None:
            raise ValueError(
                "Checkpoint is missing model_kwargs and input_dim could not be inferred. "
                "Re-train or re-save the checkpoint with model_kwargs."
            )
        model = StockPredictor(input_dim=inferred_input_dim)

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    mode = (target_mode or TARGET_MODE or "price").strip().lower()
    if mode not in {"price", "delta", "logret"}:
        mode = "price"

    return model, mode


def _yf_download_fast(
    symbol: str,
    *,
    start,
    end,
):
    """Lightweight yfinance download (no retries here; keep it snappy)."""
    import yfinance as yf

    return yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )


def _load_yf_cached_csv(cache_file: Path):
    try:
        import pandas as pd

        df = pd.read_csv(cache_file)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
        return df
    except Exception:
        return None


def _save_yf_cached_csv(df, cache_file: Path) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out = out.reset_index().rename(columns={out.index.name or "index": "Date"})
        out.to_csv(cache_file, index=False)
    except Exception:
        # Cache is best-effort.
        pass


def build_latest_features_fast(
    *,
    ticker: str,
    history_days: int,
    end_date: Optional[str],
    yf_cache_dir: Path,
    cache_ttl_hours: float,
) -> Tuple[np.ndarray, float, str]:
    """Build the latest feature vector using only price/VIX + technicals.

    - Sentiment is set to 0 (neutral)
    - No NewsAPI calls
    - No transformers pipeline
    """
    import pandas as pd
    from datetime import datetime, timedelta

    # Need enough history for indicators + the HISTORY_DAYS window.
    min_days = max(int(history_days) + 60, 180)

    end = datetime.today() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=min_days)

    cache_file = yf_cache_dir / f"{ticker.upper()}_ohlcv.csv"
    df = None
    if cache_file.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600.0
        if age_hours <= cache_ttl_hours:
            df = _load_yf_cached_csv(cache_file)

    if df is None or getattr(df, "empty", True):
        df = _yf_download_fast(ticker, start=start, end=end)
        if df is None or df.empty:
            raise ValueError(f"No yfinance data returned for {ticker}.")
        _save_yf_cached_csv(df, cache_file)

    vix_cache = yf_cache_dir / "VIX_close.csv"
    vix_df = None
    if vix_cache.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(vix_cache.stat().st_mtime)).total_seconds() / 3600.0
        if age_hours <= cache_ttl_hours:
            vix_df = _load_yf_cached_csv(vix_cache)

    if vix_df is None or getattr(vix_df, "empty", True):
        vix_df = _yf_download_fast("^VIX", start=start, end=end)
        if vix_df is not None and not vix_df.empty:
            _save_yf_cached_csv(vix_df[["Close"]], vix_cache)

    # Align and compute features to match src.data_gathering.gather_data.
    df = df.copy()
    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        df["vix_index"] = vix_df["Close"].reindex(df.index).ffill().fillna(0.0)
    else:
        df["vix_index"] = 0.0

    df["sentiment_comp"] = 0.0
    df["sentiment_global"] = 0.0

    close = df["Close"].copy()
    if hasattr(close, "iloc") and isinstance(close.iloc[0], (pd.Series, pd.DataFrame)):
        close = close.iloc[:, 0]

    df["ret_1"] = close.pct_change().fillna(0.0)
    df["logret_1"] = np.log1p(df["ret_1"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["vol_10"] = df["ret_1"].rolling(window=10, min_periods=2).std().fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=14, min_periods=2).mean()
    avg_loss = loss.rolling(window=14, min_periods=2).mean()
    rs = (avg_gain / (avg_loss.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
    df["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    df["sma_5"] = close.rolling(window=5, min_periods=2).mean().bfill().fillna(close)
    df["sma_10"] = close.rolling(window=10, min_periods=2).mean().bfill().fillna(close)
    df["ema_10"] = close.ewm(span=10, adjust=False).mean().fillna(close)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema_12 - ema_26).fillna(0.0)
    macd_signal = macd.ewm(span=9, adjust=False).mean().fillna(0.0)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = (macd - macd_signal).fillna(0.0)

    sma_20 = close.rolling(window=20, min_periods=2).mean()
    std_20 = close.rolling(window=20, min_periods=2).std().replace(0.0, np.nan)
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    df["bb_width_20"] = ((upper - lower) / np.maximum(sma_20.abs(), 1e-8)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["bb_pos_20"] = ((close - lower) / np.maximum((upper - lower), 1e-8)).replace([np.inf, -np.inf], 0.0).fillna(0.5)

    if len(df) < history_days + 2:
        raise ValueError(
            f"Not enough data to build a {history_days}-day window. Got {len(df)} rows. "
            f"Try increasing --days."
        )

    # Use the latest available day as the feature day.
    i = len(df) - 1
    start_i = i - (history_days - 1)
    window = df.iloc[start_i : i + 1][["Open", "High", "Low", "Close", "Volume"]].values.flatten()
    sentiment_vec = np.array(df.iloc[i][["sentiment_comp", "sentiment_global"]], dtype=np.float32).flatten()
    vix_value = df["vix_index"].iloc[i]
    if pd.isna(vix_value):
        vix_value = 0.0
    market_vec = np.array([vix_value], dtype=np.float32).flatten()
    tech_vec = np.array(
        df.iloc[i][
            [
                "ret_1",
                "logret_1",
                "vol_10",
                "rsi_14",
                "sma_5",
                "sma_10",
                "ema_10",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_width_20",
                "bb_pos_20",
            ]
        ],
        dtype=np.float32,
    ).flatten()

    x_last = np.concatenate([window, sentiment_vec, market_vec, tech_vec]).astype(np.float32, copy=False)

    close_val = df["Close"].iloc[i]
    if hasattr(close_val, "iloc"):
        close_val = close_val.iloc[0]
    current_close = float(close_val)
    last_date = pd.to_datetime(df.index[i]).strftime("%Y-%m-%d")
    target_date = _business_day_after(last_date)

    return x_last, current_close, target_date


def predict_next_close_pct(
    *,
    ticker: str,
    model: torch.nn.Module,
    mode: str,
    scaler_X,
    scaler_y,
    days_back: int,
    as_of: Optional[str],
    device: torch.device,
) -> Tuple[float, float, str]:
    # Build features aligned to next trading day target.
    if mode == "__with_news__":
        # Slow path: uses NewsAPI cache and transformers sentiment pipeline.
        from src.data_gathering import gather_data

        X, _, meta = gather_data(ticker, days_back=days_back, return_meta=True, end_date=as_of)
        x_last = X[-1, :].astype(np.float32, copy=False)
        current_close = float(meta["current_close"][-1])
        target_date = str(meta["target_date"][-1])
    else:
        # Fast path: no news, sentiment=0.
        x_last, current_close, target_date = build_latest_features_fast(
            ticker=ticker,
            history_days=int(getattr(model, "history_days", 0) or HISTORY_DAYS),
            end_date=as_of,
            yf_cache_dir=Path("data/raw/yf_cache"),
            cache_ttl_hours=6.0,
        )

    x_scaled = scaler_X.transform(x_last.reshape(1, -1)).astype(np.float32, copy=False)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_scaled_pred = model(x_tensor).detach().cpu().numpy().reshape(1, 1)

    y_pred = float(scaler_y.inverse_transform(y_scaled_pred).reshape(-1)[0])

    if mode == "price":
        pred_next_close = y_pred
        pct = (pred_next_close / max(current_close, 1e-8)) - 1.0
    elif mode == "delta":
        pred_next_close = current_close + y_pred
        pct = y_pred / max(current_close, 1e-8)
    else:  # logret
        pct = float(np.expm1(y_pred))
        pred_next_close = current_close * (1.0 + pct)

    return pct, pred_next_close, target_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict next trading day's close % move for a trained ticker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol (must be trained)")
    parser.add_argument("--list", action="store_true", help="List tickers that have trained artifacts")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Root folder that contains per-ticker subfolders (e.g., data/checkpoints_logret)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=max(365, HISTORY_DAYS + 60),
        help="How many days of price history to fetch for building features",
    )
    parser.add_argument(
        "--with-news",
        action="store_true",
        help="Use the full pipeline (NewsAPI + sentiment model). Slower; may hit rate limits.",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=6.0,
        help="TTL for yfinance CSV cache used by fast mode",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD) for data/news alignment (default: today)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    roots = _candidate_checkpoint_roots(args.checkpoint_root)

    if args.list:
        tickers = list_trained_tickers(roots)
        if not tickers:
            print("No trained tickers found.")
            print("Searched roots:")
            for r in roots:
                print(f"  - {r}")
            return 1
        print("Trained tickers:")
        print("  " + ", ".join(tickers))
        return 0

    ticker = (args.ticker or "").strip().upper()
    if not ticker:
        # Interactive fallback: show available tickers.
        tickers = list_trained_tickers(roots)
        if tickers:
            print("Available trained tickers:")
            print("  " + ", ".join(tickers))
        ticker = input("Enter a ticker from the trained ones: ").strip().upper()
        if not ticker:
            print("No ticker provided.")
            return 2

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    try:
        ticker_dir = resolve_ticker_dir(ticker, roots)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        available = list_trained_tickers(roots)
        if available:
            print("Available trained tickers:")
            print("  " + ", ".join(available))
        return 1

    scaler_X, scaler_y = load_scalers(ticker_dir)
    model, mode = load_model_from_checkpoint(ticker_dir, device=device)

    # Default to fast mode regardless of how it was trained.
    # If the user explicitly asks for news, switch to the slower pipeline.
    if args.with_news:
        mode = "__with_news__"

    try:
        pct, pred_next_close, target_date = predict_next_close_pct(
            ticker=ticker,
            model=model,
            mode=mode,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            days_back=args.days,
            as_of=args.as_of,
            device=device,
        )
    except Exception as e:
        # Import error type lazily; keep a safe fallback if import fails.
        try:
            from src.data_gathering import NewsAPIRateLimitError
        except Exception:
            NewsAPIRateLimitError = ()  # type: ignore

        if isinstance(e, NewsAPIRateLimitError):
            print(f"[ERROR] {e}")
            print("NewsAPI rate limit hit. Re-run later, or unset NEWS_API_KEY to force neutral sentiment.")
            return 42

        print(f"[ERROR] {e}")
        return 1

    direction = "UP" if pct >= 0 else "DOWN"
    pct_str = f"{pct * 100:+.2f}%"

    print("=" * 70)
    print("NEXT CLOSE PREDICTION")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Checkpoint: {ticker_dir}")
    print(f"Target mode: {mode if mode != '__with_news__' else 'with-news'}")
    print(f"Next trading day date: {target_date}")
    print(f"Predicted move: {pct_str} ({direction})")
    print(f"Predicted next close: {pred_next_close:.2f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

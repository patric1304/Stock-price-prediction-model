import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from transformers import pipeline
from src.config import (
    HISTORY_DAYS,
    NEWS_API_KEY,
    NEWSAPI_ENDPOINT,
    INCLUDE_GLOBAL_SENTIMENT,
    NEWS_HISTORY_DAYS,
    TARGET_MODE,
)


class NewsAPIRateLimitError(RuntimeError):
    pass

# Cache directory
CACHE_DIR = Path("data/raw/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Huggingface sentiment pipeline
_sentiment_pipeline = None
def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
    return _sentiment_pipeline

# Fetch news with caching
def fetch_newsapi_headlines(query: str, from_date: str, to_date: str, page_size: int = 20):
    cache_file = CACHE_DIR / f"{query}_{from_date}_{to_date}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if not NEWS_API_KEY:
        return []
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(NEWSAPI_ENDPOINT, params=params)
    j = resp.json()

    # Free-tier rate limiting typically returns HTTP 429 and/or code=rateLimited.
    if resp.status_code == 429 or j.get("code") == "rateLimited":
        raise NewsAPIRateLimitError(f"NewsAPI rate limit exceeded: {j}")

    if j.get("status") != "ok":
        print(f"[WARN] NewsAPI error for {query}: {j}")
        headlines = []
    else:
        headlines = [a.get("title", "") for a in j.get("articles", [])]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)
    return headlines

def compute_sentiment_score(headlines):
    if not headlines:
        return 0.0
    pipe = _get_sentiment_pipeline()
    scores = []
    for h in headlines:
        try:
            result = pipe(h)[0]
            label = result["label"].lower()
            score = result["score"]
            if "pos" in label:
                scores.append(score)
            elif "neg" in label:
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


def _yf_download_with_retry(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download OHLCV data with simple retry/backoff.

    yfinance occasionally returns empty dataframes on transient network issues
    (timeouts, rate limiting, etc.). Retrying is usually enough.
    """
    last_df: pd.DataFrame | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            last_df = df
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            # Swallow and retry; the caller will decide how to handle final failure.
            last_df = pd.DataFrame()

        if attempt < max_retries:
            time.sleep(sleep_seconds * attempt)

    return last_df if last_df is not None else pd.DataFrame()

# Gather stock + VIX + sentiment + macro features
def gather_data(
    ticker: str,
    days_back=60,
    return_meta: bool = False,
    target_mode: str | None = None,
    end_date: str | None = None,
):
    """
    Gather stock data with features (optimized for NewsAPI free tier).
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days of stock price history (default: 60)
                   News: Only last 20 days (optimized for API limits)
    
    Returns:
        X, y: Feature matrix and target values
    """
    mode = (target_mode or TARGET_MODE or "price").strip().lower()
    if mode not in {"price", "delta", "logret"}:
        raise ValueError(f"Invalid target_mode={mode!r}. Expected 'price', 'delta', or 'logret'.")

    end = datetime.today() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
    start_stock = end - timedelta(days=days_back)
    start_news = end - timedelta(days=NEWS_HISTORY_DAYS)

    df = _yf_download_with_retry(ticker, start=start_stock, end=end, max_retries=4, sleep_seconds=2.0)
    if df.empty:
        raise ValueError(
            f"No data found for {ticker}. yfinance may have timed out; try re-running."
        )

    vix = _yf_download_with_retry("^VIX", start=start_stock, end=end, max_retries=3, sleep_seconds=2.0)
    if isinstance(vix, pd.DataFrame) and (not vix.empty) and ("Close" in vix.columns):
        df["vix_index"] = vix["Close"].reindex(df.index).ffill().fillna(0.0)
    else:
        # Don't fail the whole pipeline if VIX is temporarily unavailable.
        df["vix_index"] = 0.0

    sentiments = []
    for dt in df.index:
        # Shift news sentiment by 1 day to reduce leakage from after-hours articles.
        query_dt = dt - timedelta(days=1)
        date_str = query_dt.strftime("%Y-%m-%d")

        # IMPORTANT:
        # - We only live-fetch within the last NEWS_HISTORY_DAYS to respect free-tier limits.
        # - But we still want to *use cached* headlines for older dates if they exist.
        comp_score, global_score = 0.0, 0.0

        company_cache_file = CACHE_DIR / f"{ticker}_{date_str}_{date_str}.json"
        should_try_company = company_cache_file.exists() or (query_dt >= start_news)
        if should_try_company:
            try:
                company_news = fetch_newsapi_headlines(ticker, date_str, date_str)
                comp_score = compute_sentiment_score(company_news)
            except NewsAPIRateLimitError:
                raise
            except Exception:
                comp_score = 0.0

        if INCLUDE_GLOBAL_SENTIMENT:
            global_cache_file = CACHE_DIR / f"global economy_{date_str}_{date_str}.json"
            should_try_global = global_cache_file.exists() or (query_dt >= start_news)
            if should_try_global:
                try:
                    global_news = fetch_newsapi_headlines("global economy", date_str, date_str)
                    global_score = compute_sentiment_score(global_news)
                except NewsAPIRateLimitError:
                    raise
                except Exception:
                    global_score = 0.0

        sentiments.append((comp_score, global_score))
    df["sentiment_comp"] = [s[0] for s in sentiments]
    df["sentiment_global"] = [s[1] for s in sentiments]

    # -------------------------------
    # Leak-safe technical indicators
    # -------------------------------
    # All indicators are computed from values available up to the same day index.
    close = df["Close"].copy()
    if hasattr(close, "iloc") and isinstance(close.iloc[0], (pd.Series, pd.DataFrame)):
        # yfinance sometimes returns multi-index columns even for a single ticker.
        close = close.iloc[:, 0]

    # Returns
    df["ret_1"] = close.pct_change().fillna(0.0)
    df["logret_1"] = np.log1p(df["ret_1"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Volatility (rolling std of daily returns)
    df["vol_10"] = df["ret_1"].rolling(window=10, min_periods=2).std().fillna(0.0)

    # RSI (14)
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=14, min_periods=2).mean()
    avg_loss = loss.rolling(window=14, min_periods=2).mean()
    rs = (avg_gain / (avg_loss.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
    df["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    # Moving averages / EMA
    df["sma_5"] = close.rolling(window=5, min_periods=2).mean().bfill().fillna(close)
    df["sma_10"] = close.rolling(window=10, min_periods=2).mean().bfill().fillna(close)
    df["ema_10"] = close.ewm(span=10, adjust=False).mean().fillna(close)

    # MACD (12, 26) + signal (9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema_12 - ema_26).fillna(0.0)
    macd_signal = macd.ewm(span=9, adjust=False).mean().fillna(0.0)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = (macd - macd_signal).fillna(0.0)

    # Bollinger Bands (20) - normalized position and band width
    sma_20 = close.rolling(window=20, min_periods=2).mean()
    std_20 = close.rolling(window=20, min_periods=2).std().replace(0.0, np.nan)
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    df["bb_width_20"] = ((upper - lower) / np.maximum(sma_20.abs(), 1e-8)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["bb_pos_20"] = ((close - lower) / np.maximum((upper - lower), 1e-8)).replace([np.inf, -np.inf], 0.0).fillna(0.5)

    # Build dataset
    X, y = [], []
    meta = {
        "current_close": [],
        "target_date": [],
    }

    def _close_scalar(idx: int) -> float:
        close_val = df["Close"].iloc[idx]
        # With yfinance multi-index columns, this can be a Series (even for one ticker)
        if hasattr(close_val, "iloc"):
            close_val = close_val.iloc[0]
        return float(close_val)

    # Build samples for predicting next-day close (i+1) using information available at day i.
    # Include day i in the OHLCV window so the model has access to the current close, matching the naive baseline.
    for i in range(HISTORY_DAYS - 1, len(df) - 1):
        start = i - (HISTORY_DAYS - 1)
        window = df.iloc[start : i + 1][["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        sentiment_vec = np.array(df.iloc[i][["sentiment_comp","sentiment_global"]], dtype=np.float32).flatten()
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
        X_i = np.concatenate([window, sentiment_vec, market_vec, tech_vec])

        # Target options:
        # - price:  next-day close
        # - delta:  next-day change relative to today's close
        # - logret: next-day log return ln(close[t+1] / close[t])
        next_close = _close_scalar(i + 1)
        current_close = _close_scalar(i)

        if mode == "price":
            y_val = next_close
        elif mode == "delta":
            y_val = (next_close - current_close)
        else:
            # Log return. Prices should be positive; guard anyway.
            denom = current_close if current_close > 0 else 1e-8
            y_val = float(np.log(max(next_close, 1e-8) / denom))

        # Always store target as shape (1,) so y becomes (N, 1) after np.array
        y_i = np.float32(y_val)
        X.append(X_i)
        y.append([y_i])

        if return_meta:
            meta["current_close"].append(current_close)
            meta["target_date"].append(df.index[i + 1].strftime("%Y-%m-%d"))

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)  # (N, 1)

    if return_meta:
        meta["current_close"] = np.array(meta["current_close"], dtype=np.float32)
        meta["target_date"] = np.array(meta["target_date"], dtype=object)
        return X_arr, y_arr, meta
    return X_arr, y_arr

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.config import (
    HISTORY_DAYS,
    INCLUDE_GLOBAL_SENTIMENT,
    NEWS_API_KEY,
    NEWSAPI_ENDPOINT,
    NEWS_HISTORY_DAYS,
    TARGET_MODE,
)


class NewsAPIRateLimitError(RuntimeError):
    pass


CACHE_DIR = Path("data/raw/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


_sentiment_pipeline = None


def _get_sentiment_pipeline():
    # Import transformers lazily so training that doesn't use recent news stays fast.
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f",
        )
    return _sentiment_pipeline


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
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
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
            last_df = pd.DataFrame()

        if attempt < max_retries:
            time.sleep(sleep_seconds * attempt)

    return last_df if last_df is not None else pd.DataFrame()


@dataclass
class GatherMeta:
    current_close: np.ndarray
    target_date: np.ndarray


def gather_data_deep_experiment(
    ticker: str,
    *,
    days_back: int = 1825,
    news_history_days: Optional[int] = None,
    strict_news_cutoff: bool = True,
    return_meta: bool = False,
    target_mode: str | None = None,
    end_date: str | None = None,
    history_days: int | None = None,
):
    """Deep-experiment data gatherer.

    Key behavior vs src.data_gathering.gather_data:
    - Uses 5y by default.
    - Sentiment is forced to 0 before the last `news_history_days`.
      (Even if cached headlines exist.)
    - transformers sentiment pipeline is imported lazily.
    """

    mode = (target_mode or TARGET_MODE or "price").strip().lower()
    if mode not in {"price", "delta", "logret"}:
        raise ValueError(f"Invalid target_mode={mode!r}. Expected 'price', 'delta', or 'logret'.")

    nhd = int(NEWS_HISTORY_DAYS if news_history_days is None else news_history_days)

    end = datetime.today() if not end_date else datetime.strptime(end_date, "%Y-%m-%d")
    start_stock = end - timedelta(days=int(days_back))
    start_news = end - timedelta(days=int(nhd))

    df = _yf_download_with_retry(ticker, start=start_stock, end=end, max_retries=4, sleep_seconds=2.0)
    if df.empty:
        raise ValueError(f"No data found for {ticker}. yfinance may have timed out; try re-running.")

    vix = _yf_download_with_retry("^VIX", start=start_stock, end=end, max_retries=3, sleep_seconds=2.0)
    if isinstance(vix, pd.DataFrame) and (not vix.empty) and ("Close" in vix.columns):
        df["vix_index"] = vix["Close"].reindex(df.index).ffill().fillna(0.0)
    else:
        df["vix_index"] = 0.0

    sentiments = []
    for dt in df.index:
        query_dt = dt - timedelta(days=1)
        date_str = query_dt.strftime("%Y-%m-%d")

        comp_score, global_score = 0.0, 0.0

        if (not strict_news_cutoff) or (query_dt >= start_news):
            try:
                company_news = fetch_newsapi_headlines(ticker, date_str, date_str)
                comp_score = compute_sentiment_score(company_news)
            except NewsAPIRateLimitError:
                raise
            except Exception:
                comp_score = 0.0

            if INCLUDE_GLOBAL_SENTIMENT:
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

    X, y = [], []
    meta_current_close = []
    meta_target_date = []

    def _close_scalar(idx: int) -> float:
        close_val = df["Close"].iloc[idx]
        if hasattr(close_val, "iloc"):
            close_val = close_val.iloc[0]
        return float(close_val)

    hd = int(HISTORY_DAYS if history_days is None else history_days)
    if hd <= 1:
        raise ValueError(f"history_days must be >= 2, got {hd}")

    for i in range(hd - 1, len(df) - 1):
        start_i = i - (hd - 1)
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

        X_i = np.concatenate([window, sentiment_vec, market_vec, tech_vec])

        next_close = _close_scalar(i + 1)
        current_close = _close_scalar(i)

        if mode == "price":
            y_val = next_close
        elif mode == "delta":
            y_val = (next_close - current_close)
        else:
            denom = current_close if current_close > 0 else 1e-8
            y_val = float(np.log(max(next_close, 1e-8) / denom))

        X.append(X_i.astype(np.float32, copy=False))
        y.append([np.float32(y_val)])

        if return_meta:
            meta_current_close.append(current_close)
            meta_target_date.append(df.index[i + 1].strftime("%Y-%m-%d"))

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    if return_meta:
        meta = {
            "current_close": np.array(meta_current_close, dtype=np.float32),
            "target_date": np.array(meta_target_date, dtype=object),
        }
        return X_arr, y_arr, meta

    return X_arr, y_arr

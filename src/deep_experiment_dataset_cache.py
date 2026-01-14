from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DeepDatasetCacheKey:
    ticker: str
    as_of: str
    days_back: int
    history_days: int
    target_mode: str
    news_days: int


def _safe_token(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in {"-", "_"})


def make_cache_path(cache_dir: Path, key: DeepDatasetCacheKey) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"{key.ticker.upper()}"
        f"__asof-{_safe_token(key.as_of)}"
        f"__days-{int(key.days_back)}"
        f"__hist-{int(key.history_days)}"
        f"__mode-{_safe_token(key.target_mode)}"
        f"__news-{int(key.news_days)}"
        ".parquet"
    )
    return cache_dir / fname


def save_dataset_parquet(
    *,
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    current_close: Optional[np.ndarray] = None,
    target_date: Optional[np.ndarray] = None,
) -> None:
    """Save dataset to Parquet.

    Requires `pyarrow` installed (recommended).
    """
    import pandas as pd

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(y) != X.shape[0]:
        raise ValueError(f"y length {len(y)} != X rows {X.shape[0]}")

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["y"] = y

    if current_close is not None:
        cc = np.asarray(current_close).reshape(-1)
        if len(cc) != len(df):
            raise ValueError(f"current_close length {len(cc)} != rows {len(df)}")
        df["current_close"] = cc

    if target_date is not None:
        td = np.asarray(target_date).reshape(-1)
        if len(td) != len(df):
            raise ValueError(f"target_date length {len(td)} != rows {len(df)}")
        df["target_date"] = td.astype(str)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_dataset_parquet(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import pandas as pd

    df = pd.read_parquet(path)

    feature_cols = [c for c in df.columns if c.startswith("f")]
    feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]))

    if "y" not in df.columns:
        raise ValueError("Parquet dataset missing required column 'y'")

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["y"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)

    meta: dict = {}
    if "current_close" in df.columns:
        meta["current_close"] = df["current_close"].to_numpy(dtype=np.float32, copy=False)
    if "target_date" in df.columns:
        meta["target_date"] = df["target_date"].astype(str).to_numpy(dtype=object, copy=False)

    return X, y, meta


def effective_as_of_label(as_of: Optional[str]) -> str:
    return as_of if as_of else datetime.today().strftime("%Y-%m-%d")

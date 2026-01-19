"""Generate presentation-ready plots into a dedicated folder.

Creates a small set of clean plots for your teacher presentation:
1) Architecture diagram (multimodal + BiLSTM + Attention + MLP)
2) Metrics comparison (Baseline vs Wide vs Deep)
3) Direction timeline heatmap (Actual vs Wide vs Deep)
4) Confusion matrix (Deep model)

Designed to reuse existing evaluation utilities and (when available) the deep
Parquet dataset cache to avoid re-downloading data / hitting NewsAPI.

Examples:
  python scripts/generate_presentation_plots.py
  python scripts/generate_presentation_plots.py --ticker AAPL --as-of 2026-01-14
  python scripts/generate_presentation_plots.py --ticker MSFT --window 120

Outputs land in: presentation_assets/plots/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PLOT_DPI = 300


def _apply_presentation_style() -> None:
    """Apply a clean, presentation-friendly plot theme.

    Kept local to this script so it doesn't affect other plotting code.
    """
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")


def _annotate_bars(ax, fmt: str = "{:.2f}") -> None:
    """Annotate bars with their heights."""
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


@dataclass(frozen=True)
class EvalArtifacts:
    metrics: dict
    baseline: dict
    dates: list[str]
    current_close: np.ndarray


def _try_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_as_of_from_training_report(ticker_dir: Path) -> Optional[str]:
    tr = _try_read_json(ticker_dir / "training_report.json")
    if not tr:
        return None

    cfg = (tr.get("configuration") or {})
    as_of = cfg.get("as_of")
    if isinstance(as_of, str) and len(as_of) >= 10:
        return as_of[:10]

    ts = tr.get("timestamp")
    if isinstance(ts, str) and len(ts) >= 10:
        return ts[:10]

    return None


def _pick_latest_deep_cache_as_of(
    cache_dir: Path,
    *,
    ticker: str,
    days_back: int,
    history_days: int,
    target_mode: str,
    news_days: int,
) -> Optional[str]:
    """Pick the newest as_of in deep parquet cache matching parameters."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None

    # Example filename:
    # AAPL__asof-2026-01-14__days-1825__hist-30__mode-logret__news-30.parquet
    pattern = re.compile(
        rf"^{re.escape(ticker.upper())}__asof-(?P<asof>[^_]+)__days-{int(days_back)}__hist-{int(history_days)}__mode-{re.escape(target_mode)}__news-{int(news_days)}\\.parquet$"
    )

    as_of_candidates: list[str] = []
    for p in cache_dir.glob(f"{ticker.upper()}__asof-*.parquet"):
        m = pattern.match(p.name)
        if m:
            as_of_candidates.append(m.group("asof"))

    if not as_of_candidates:
        return None

    # Dates are ISO like YYYY-MM-DD; string max works.
    return sorted(as_of_candidates)[-1]


def _load_deep_dataset_from_cache(
    *,
    cache_dir: Path,
    ticker: str,
    as_of: Optional[str],
    days_back: int,
    history_days: int,
    target_mode: str,
    news_days: int,
):
    from src.deep_experiment_dataset_cache import (
        DeepDatasetCacheKey,
        effective_as_of_label,
        load_dataset_parquet,
        make_cache_path,
    )

    cache_dir = Path(cache_dir)

    resolved_as_of = effective_as_of_label(as_of)
    key = DeepDatasetCacheKey(
        ticker=ticker.upper(),
        as_of=resolved_as_of,
        days_back=int(days_back),
        history_days=int(history_days),
        target_mode=str(target_mode),
        news_days=int(news_days),
    )
    path = make_cache_path(cache_dir, key)

    if not path.exists():
        # Try newest cache if exact as_of not present
        newest = _pick_latest_deep_cache_as_of(
            cache_dir,
            ticker=ticker,
            days_back=days_back,
            history_days=history_days,
            target_mode=target_mode,
            news_days=news_days,
        )
        if newest:
            key = DeepDatasetCacheKey(
                ticker=ticker.upper(),
                as_of=str(newest),
                days_back=int(days_back),
                history_days=int(history_days),
                target_mode=str(target_mode),
                news_days=int(news_days),
            )
            path = make_cache_path(cache_dir, key)

    if not path.exists():
        raise FileNotFoundError(
            f"Deep dataset cache not found. Looked for: {path}. "
            f"Available files are in: {cache_dir}"
        )

    X, y, meta = load_dataset_parquet(path)
    if "current_close" not in meta:
        raise ValueError(f"Deep cache missing current_close: {path}")
    if "target_date" not in meta:
        # We can still plot with index labels, but dates are nicer.
        meta["target_date"] = np.array(["" for _ in range(len(X))], dtype=object)

    return X, y, meta


def _read_checkpoint_metadata(model_path: Path) -> tuple[Optional[int], Optional[int], str]:
    """Return (history_days, input_dim, target_mode)."""
    import torch

    history_days = None
    input_dim = None
    target_mode = "logret"

    try:
        ck = torch.load(str(model_path), map_location="cpu")
        if isinstance(ck, dict):
            tm = ck.get("target_mode")
            if isinstance(tm, str) and tm.strip().lower() in {"price", "delta", "logret"}:
                target_mode = tm.strip().lower()

            mk = ck.get("model_kwargs")
            if isinstance(mk, dict):
                hd = mk.get("history_days")
                if hd is not None:
                    history_days = int(hd)
                inp = mk.get("input_dim")
                if inp is not None:
                    input_dim = int(inp)
    except Exception:
        pass

    return history_days, input_dim, target_mode


def _evaluate_from_checkpoint(
    *,
    checkpoint_root: Path,
    ticker: str,
    X: np.ndarray,
    y: np.ndarray,
    current_close: np.ndarray,
    target_date: Optional[np.ndarray],
    device: str,
    force_target_mode: Optional[str] = None,
) -> EvalArtifacts:
    from scripts.evaluate_advanced_model import (
        evaluate_baseline_naive,
        evaluate_model_comprehensive,
        load_model_and_scalers,
    )

    ticker_dir = Path(checkpoint_root) / ticker.upper()
    model_path = ticker_dir / "best_model.pth"
    scaler_x_path = ticker_dir / "scaler_X.pkl"
    scaler_y_path = ticker_dir / "scaler_y.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not scaler_x_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError(f"Missing scalers in: {ticker_dir}")

    _, ck_input_dim, ck_target_mode = _read_checkpoint_metadata(model_path)

    if ck_input_dim is not None and int(ck_input_dim) != int(X.shape[1]):
        raise ValueError(
            f"Feature dim mismatch for {ticker}: checkpoint expects {ck_input_dim}, got {X.shape[1]}."
        )

    model, scaler_X, scaler_y, loaded_target_mode = load_model_and_scalers(
        model_path=str(model_path),
        scaler_X_path=str(scaler_x_path),
        scaler_y_path=str(scaler_y_path),
        input_dim=int(ck_input_dim) if ck_input_dim is not None else int(X.shape[1]),
        device=device,
    )

    desired_mode = force_target_mode or loaded_target_mode or ck_target_mode or "logret"
    desired_mode = str(desired_mode).strip().lower()
    if desired_mode not in {"price", "delta", "logret"}:
        desired_mode = "logret"

    test_size = max(10, int(0.15 * len(X)))
    X_test = X[-test_size:]
    y_test = y[-test_size:].reshape(-1)
    cc_test = np.asarray(current_close, dtype=np.float32).reshape(-1)[-test_size:]

    metrics = evaluate_model_comprehensive(
        model,
        X_test,
        y_test,
        scaler_X,
        scaler_y,
        device=device,
        current_close=cc_test,
        target_mode=desired_mode,
    )

    baseline = evaluate_baseline_naive(
        y_true=np.asarray(metrics["targets"], dtype=np.float32),
        current_close=cc_test,
        target_mode=metrics.get("target_mode") or desired_mode,
    )

    if target_date is None:
        dates = [str(i) for i in range(len(X_test))]
    else:
        td = np.asarray(target_date).reshape(-1)[-test_size:]
        dates = [str(d) for d in td]

    return EvalArtifacts(metrics=metrics, baseline=baseline, dates=dates, current_close=cc_test)


def _direction_signals_by_date(art: EvalArtifacts) -> dict[str, tuple[int, int]]:
    """Map date -> (actual_sign, pred_sign) using the same definition as evaluation.

    actual_sign: sign(target_price - current_close)
    pred_sign:   sign(pred_price - current_close)
    Encoded as: +1 (up), -1 (down)
    """

    preds = np.asarray(art.metrics["predictions"], dtype=np.float32).reshape(-1)
    targs = np.asarray(art.metrics["targets"], dtype=np.float32).reshape(-1)
    cc = np.asarray(art.current_close, dtype=np.float32).reshape(-1)
    dates = list(art.dates)

    n = min(len(preds), len(targs), len(cc), len(dates))
    preds = preds[-n:]
    targs = targs[-n:]
    cc = cc[-n:]
    dates = dates[-n:]

    actual_up = (targs - cc) > 0
    pred_up = (preds - cc) > 0

    out: dict[str, tuple[int, int]] = {}
    for d, a, p in zip(dates, actual_up, pred_up, strict=False):
        if not d:
            continue
        out[str(d)] = (1 if bool(a) else -1, 1 if bool(p) else -1)
    return out


def _gather_standard_dataset(
    *,
    ticker: str,
    days_back: int,
    end_date: Optional[str],
    history_days: Optional[int],
    target_mode: str,
):
    from src.data_gathering import gather_data

    X, y, meta = gather_data(
        ticker,
        days_back=int(days_back),
        return_meta=True,
        target_mode=target_mode,
        end_date=end_date,
        history_days=history_days,
    )
    return X, y, meta


def _plot_architecture(out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _apply_presentation_style()

    fig = plt.figure(figsize=(12.8, 5.2))
    ax = fig.add_subplot(111)
    ax.axis("off")

    def box(x, y, w, h, text, *, fc: str, ec: str):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=2,
                edgecolor=ec,
                facecolor=fc,
                alpha=0.95,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=2, color="#333333"),
        )

    # Palette (safe, readable)
    c_input = "#E8F0FE"
    c_fuse = "#E6F4EA"
    c_seq = "#FEF7E0"
    c_head = "#FCE8E6"
    c_edge = "#333333"

    # Inputs
    box(0.05, 0.62, 0.22, 0.25, "Market Data\nOHLCV + VIX\n+ Indicators", fc=c_input, ec=c_edge)
    box(0.05, 0.13, 0.22, 0.25, "News Data\nArticles → FinBERT\nSentiment", fc=c_input, ec=c_edge)

    # Fusion
    box(0.33, 0.38, 0.18, 0.25, "Feature\nFusion", fc=c_fuse, ec=c_edge)

    # Sequence model
    box(0.56, 0.38, 0.16, 0.25, "BiLSTM\n(Sequence)", fc=c_seq, ec=c_edge)
    box(0.76, 0.38, 0.16, 0.25, "Attention\n(Focus)", fc=c_seq, ec=c_edge)

    # Head
    box(0.76, 0.10, 0.16, 0.18, "Residual\nMLP", fc=c_head, ec=c_edge)
    box(0.56, 0.10, 0.16, 0.18, "Output\nReturn / Direction", fc=c_head, ec=c_edge)

    # Arrows
    arrow(0.27, 0.75, 0.33, 0.50)
    arrow(0.27, 0.25, 0.33, 0.50)
    arrow(0.51, 0.50, 0.56, 0.50)
    arrow(0.72, 0.50, 0.76, 0.50)
    arrow(0.84, 0.38, 0.84, 0.28)
    arrow(0.76, 0.19, 0.72, 0.19)

    ax.set_title("Multimodal Architecture (Market + News)")

    _save_fig(fig, out_path)
    plt.close(fig)


def _plot_metrics_comparison(
    *,
    ticker: str,
    wide: EvalArtifacts,
    deep: EvalArtifacts,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _apply_presentation_style()

    labels = [
        "Baseline (Wide)",
        "Wide Model",
        "Baseline (Deep)",
        "Deep Model",
    ]
    rmse_vals = [
        float(wide.baseline.get("rmse", np.nan)),
        float(wide.metrics.get("rmse", np.nan)),
        float(deep.baseline.get("rmse", np.nan)),
        float(deep.metrics.get("rmse", np.nan)),
    ]
    dir_vals = [
        float(wide.baseline.get("directional_accuracy", np.nan)),
        float(wide.metrics.get("directional_accuracy", np.nan)),
        float(deep.baseline.get("directional_accuracy", np.nan)),
        float(deep.metrics.get("directional_accuracy", np.nan)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # Colors: group wide vs deep
    colors = ["#AECBFA", "#669DF6", "#CDE7D8", "#34A853"]

    ax1.bar(labels, rmse_vals, color=colors, edgecolor="#333333", linewidth=0.8)
    ax1.set_title("RMSE (Lower is better)")
    ax1.set_ylabel("RMSE")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)
    _annotate_bars(ax1, fmt="{:.3f}")

    ax2.bar(labels, dir_vals, color=colors, edgecolor="#333333", linewidth=0.8)
    ax2.axhline(50, linestyle="--", linewidth=1.5, color="#666666", alpha=0.8)
    ax2.set_title("Directional Accuracy (Higher is better)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, max(100.0, float(np.nanmax(dir_vals)) + 5.0))
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, axis="y", alpha=0.25)
    _annotate_bars(ax2, fmt="{:.1f}%")

    fig.suptitle(f"{ticker} — Baseline vs Wide vs Deep")
    fig.tight_layout()

    _save_fig(fig, out_path)

    plt.close(fig)


def _plot_direction_timeline_heatmap(
    *,
    ticker: str,
    wide: EvalArtifacts,
    deep: EvalArtifacts,
    window: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    _apply_presentation_style()

    wide_map = _direction_signals_by_date(wide)
    deep_map = _direction_signals_by_date(deep)

    common_dates = sorted(set(wide_map.keys()) & set(deep_map.keys()))
    if len(common_dates) <= 5:
        raise ValueError(
            "Not enough overlapping dates between Wide and Deep to plot timeline. "
            "Try using the same --as-of, or re-generate deep cache for that date."
        )

    common_dates = common_dates[-int(window) :]

    actual = np.array([deep_map[d][0] for d in common_dates], dtype=np.int8)
    wide_pred = np.array([wide_map[d][1] for d in common_dates], dtype=np.int8)
    deep_pred = np.array([deep_map[d][1] for d in common_dates], dtype=np.int8)

    signals = np.vstack([actual, wide_pred, deep_pred])
    ylabels = ["Actual (from deep)", "Wide Pred", "Deep Pred"]

    fig = plt.figure(figsize=(14.5, 3.2))
    ax = fig.add_subplot(111)

    # Discrete colormap: Down (red) / Up (green)
    # signals are -1 or +1; map to {0,1}
    heat = ((signals + 1) // 2).astype(int)
    cmap = ListedColormap(["#EA4335", "#34A853"])

    sns.heatmap(
        heat,
        cmap=cmap,
        cbar=True,
        cbar_kws={"ticks": [0.25, 0.75], "shrink": 0.75},
        yticklabels=ylabels,
        xticklabels=False,
        linewidths=0.5,
        linecolor="#FFFFFF",
        ax=ax,
    )

    # Label colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(["Down", "Up"])

    # Add sparse date ticks (about 8 labels)
    tick_count = 8
    idxs = np.linspace(0, len(common_dates) - 1, tick_count).astype(int)
    ax.set_xticks([i + 0.5 for i in idxs])
    ax.set_xticklabels([common_dates[i] for i in idxs], rotation=30, ha="right", fontsize=8)

    ax.set_title(f"{ticker} — Direction Timeline (Green=Up, Red=Down)")
    ax.set_xlabel("Date")

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def _plot_confusion_matrix(
    *,
    ticker: str,
    art: EvalArtifacts,
    out_path: Path,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    _apply_presentation_style()

    preds = np.asarray(art.metrics["predictions"], dtype=np.float32).reshape(-1)
    targs = np.asarray(art.metrics["targets"], dtype=np.float32).reshape(-1)
    cc = np.asarray(art.current_close, dtype=np.float32).reshape(-1)

    n = min(len(preds), len(targs), len(cc))
    preds = preds[-n:]
    targs = targs[-n:]
    cc = cc[-n:]

    # Same definition as directional_accuracy in evaluation.
    y_true = ((targs - cc) > 0).astype(int)
    y_pred = ((preds - cc) > 0).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Build annotations: count + percentage
    total = float(np.sum(cm)) if np.sum(cm) > 0 else 1.0
    cm_pct = (cm / total) * 100.0
    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"

    acc = float(np.mean(y_true == y_pred) * 100.0) if len(y_true) else 0.0

    fig = plt.figure(figsize=(5.4, 4.8))
    ax = fig.add_subplot(111)

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        cbar=False,
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"],
        square=True,
        linewidths=0.8,
        linecolor="#FFFFFF",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{ticker} — Confusion Matrix ({title_suffix})\nAccuracy: {acc:.1f}%")

    fig.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate presentation plots into presentation_assets/plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to showcase")
    parser.add_argument("--as-of", type=str, default=None, help="End date YYYY-MM-DD (tries to auto-infer)")
    parser.add_argument("--days", type=int, default=1825, help="Days of history to evaluate")
    parser.add_argument("--window", type=int, default=140, help="How many test samples to show in timeline")

    parser.add_argument("--wide-root", type=str, default="data/checkpoints_logret", help="Wide/old checkpoints root")
    parser.add_argument("--deep-root", type=str, default="data/checkpoints_deep_5y", help="Deep checkpoints root")

    parser.add_argument(
        "--deep-cache-dir",
        type=str,
        default="data/processed/deep_experiment_datasets",
        help="Deep Parquet cache directory",
    )
    parser.add_argument("--deep-news-days", type=int, default=30, help="Deep cache key: news days")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="presentation_assets/plots",
        help="Output folder for plots",
    )

    args = parser.parse_args()

    # Lazy imports so `--help` is fast.
    import torch

    ticker = args.ticker.upper()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wide_root = PROJECT_ROOT / args.wide_root
    deep_root = PROJECT_ROOT / args.deep_root

    # Try to infer as-of date if user didn’t provide one
    as_of = args.as_of
    if as_of is None:
        deep_asof = _infer_as_of_from_training_report(deep_root / ticker)
        wide_asof = _infer_as_of_from_training_report(wide_root / ticker)
        as_of = deep_asof or wide_asof

    # Read checkpoint metadata to match feature window
    deep_history_days, _, deep_target_mode = _read_checkpoint_metadata(deep_root / ticker / "best_model.pth")
    wide_history_days, _, wide_target_mode = _read_checkpoint_metadata(wide_root / ticker / "best_model.pth")

    # Fallbacks
    deep_history_days = int(deep_history_days) if deep_history_days is not None else 30
    wide_history_days = int(wide_history_days) if wide_history_days is not None else 30

    # Load datasets
    X_deep, y_deep, meta_deep = _load_deep_dataset_from_cache(
        cache_dir=PROJECT_ROOT / args.deep_cache_dir,
        ticker=ticker,
        as_of=as_of,
        days_back=int(args.days),
        history_days=int(deep_history_days),
        target_mode=str(deep_target_mode),
        news_days=int(args.deep_news_days),
    )

    X_wide, y_wide, meta_wide = _gather_standard_dataset(
        ticker=ticker,
        days_back=int(args.days),
        end_date=as_of,
        history_days=int(wide_history_days),
        target_mode=str(wide_target_mode),
    )

    # Evaluate
    deep_eval = _evaluate_from_checkpoint(
        checkpoint_root=deep_root,
        ticker=ticker,
        X=X_deep,
        y=y_deep,
        current_close=np.asarray(meta_deep["current_close"]),
        target_date=np.asarray(meta_deep.get("target_date")) if meta_deep.get("target_date") is not None else None,
        device=device,
        force_target_mode=str(deep_target_mode),
    )

    wide_eval = _evaluate_from_checkpoint(
        checkpoint_root=wide_root,
        ticker=ticker,
        X=X_wide,
        y=y_wide,
        current_close=np.asarray(meta_wide["current_close"]),
        target_date=np.asarray(meta_wide.get("target_date")) if meta_wide.get("target_date") is not None else None,
        device=device,
        force_target_mode=str(wide_target_mode),
    )

    # Plots
    _plot_architecture(out_dir / "architecture_diagram.png")
    _plot_metrics_comparison(
        ticker=ticker,
        wide=wide_eval,
        deep=deep_eval,
        out_path=out_dir / f"metrics_comparison_{ticker}.png",
    )
    _plot_direction_timeline_heatmap(
        ticker=ticker,
        wide=wide_eval,
        deep=deep_eval,
        window=int(args.window),
        out_path=out_dir / f"direction_timeline_{ticker}.png",
    )
    _plot_confusion_matrix(
        ticker=ticker,
        art=deep_eval,
        out_path=out_dir / f"confusion_matrix_deep_{ticker}.png",
        title_suffix="Deep",
    )

    # Also drop a small JSON summary for quick copy/paste into slides
    summary = {
        "ticker": ticker,
        "as_of": as_of,
        "wide": {
            "rmse": float(wide_eval.metrics.get("rmse", np.nan)),
            "directional_accuracy": float(wide_eval.metrics.get("directional_accuracy", np.nan)),
            "baseline": wide_eval.baseline,
        },
        "deep": {
            "rmse": float(deep_eval.metrics.get("rmse", np.nan)),
            "directional_accuracy": float(deep_eval.metrics.get("directional_accuracy", np.nan)),
            "baseline": deep_eval.baseline,
        },
        "plots": {
            "architecture": str((out_dir / "architecture_diagram.png").as_posix()),
            "metrics_comparison": str((out_dir / f"metrics_comparison_{ticker}.png").as_posix()),
            "direction_timeline": str((out_dir / f"direction_timeline_{ticker}.png").as_posix()),
            "confusion_matrix_deep": str((out_dir / f"confusion_matrix_deep_{ticker}.png").as_posix()),
        },
    }
    with open(out_dir / f"presentation_summary_{ticker}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved plots to: {out_dir}")
    print(f"[OK] Saved summary to: {out_dir / f'presentation_summary_{ticker}.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a professor-friendly PDF report for this project.

Creates:
- docs/Stock_Price_Prediction_Report.pdf

The report includes:
- End-to-end pipeline explanation
- File-by-file walkthrough
- Key code snippets (excerpts) from the current workspace files
- Presentation script + slide outline + common Q&A

Run:
  python scripts/generate_project_report_pdf.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
OUT_PDF = DOCS_DIR / "Stock_Price_Prediction_Report.pdf"


@dataclass
class Snippet:
    title: str
    path: Path
    text: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def clip_lines(text: str, max_lines: int = 120) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.strip("\n")
    head = lines[:max_lines]
    head.append("# ... (truncated for report) ...")
    return "\n".join(head).strip("\n")


def extract_block(text: str, start_pat: str, end_pat: str, flags: int = re.S) -> Optional[str]:
    m1 = re.search(start_pat, text, flags)
    if not m1:
        return None
    start = m1.start()
    m2 = re.search(end_pat, text[m1.end() :], flags)
    if not m2:
        return None
    end = m1.end() + m2.end()
    return text[start:end]


def extract_function(text: str, func_name: str) -> Optional[str]:
                                                                                   
    pat = rf"^def\s+{re.escape(func_name)}\s*\(.*?\):\n"
    m = re.search(pat, text, flags=re.M)
    if not m:
        return None
    start = m.start()
    tail = text[m.end() :]
    m2 = re.search(r"^(def|class)\s+", tail, flags=re.M)
    end = (m.end() + m2.start()) if m2 else len(text)
    return text[start:end]


def extract_class_method(text: str, class_name: str, method_name: str) -> Optional[str]:
                                                            
    class_pat = rf"^class\s+{re.escape(class_name)}\b.*?:\n"
    mc = re.search(class_pat, text, flags=re.M)
    if not mc:
        return None
    class_start = mc.start()
    class_tail = text[mc.end() :]
                                             
    m2 = re.search(r"^(class|def)\s+", class_tail, flags=re.M)
    class_end = (mc.end() + m2.start()) if m2 else len(text)
    class_block = text[class_start:class_end]

                 
    meth_pat = rf"^\s+def\s+{re.escape(method_name)}\s*\(.*?\):\n"
    mm = re.search(meth_pat, class_block, flags=re.M)
    if not mm:
        return None
    start = mm.start()
    tail = class_block[mm.end() :]
    m3 = re.search(r"^\s+def\s+", tail, flags=re.M)
    end = (mm.end() + m3.start()) if m3 else len(class_block)
    return class_block[start:end]


def make_snippets() -> list[Snippet]:
    snippets: list[Snippet] = []

                
    config_py = ROOT / "src" / "config.py"
    data_gathering_py = ROOT / "src" / "data_gathering.py"
    model_py = ROOT / "src" / "model.py"
    train_py = ROOT / "src" / "train.py"
    preprocessing_py = ROOT / "src" / "preprocessing.py"
    dataset_py = ROOT / "src" / "dataset.py"

    eval_py = ROOT / "scripts" / "evaluate_advanced_model.py"
    train_script_py = ROOT / "scripts" / "train_advanced_model.py"
    pipeline_py = ROOT / "scripts" / "run_pipeline.py"

               
    snippets.append(
        Snippet(
            title="Configuration (target + history window)",
            path=config_py,
            text=clip_lines(read_text(config_py), 120),
        )
    )

                    
    dg_txt = read_text(data_gathering_py)
    gd = extract_function(dg_txt, "gather_data") or dg_txt
    snippets.append(
        Snippet(
            title="Dataset construction: gather_data() (features + targets)",
            path=data_gathering_py,
            text=clip_lines(gd, 160),
        )
    )

                
    snippets.append(
        Snippet(
            title="Scaling (StandardScaler)",
            path=preprocessing_py,
            text=clip_lines(read_text(preprocessing_py), 80),
        )
    )

                        
    snippets.append(
        Snippet(
            title="PyTorch dataset wrapper", path=dataset_py, text=clip_lines(read_text(dataset_py), 80)
        )
    )

                      
    m_txt = read_text(model_py)
    forward = extract_class_method(m_txt, "AdvancedStockPredictor", "forward") or ""
    snippets.append(
        Snippet(
            title="Model forward() (reshape window -> sequence -> LSTM -> attention)",
            path=model_py,
            text=clip_lines(forward or m_txt, 140),
        )
    )

                             
    t_txt = read_text(train_py)
    tma = extract_function(t_txt, "train_model_advanced") or t_txt
    snippets.append(
        Snippet(
            title="Training loop (split, scaling, early stopping, checkpoint)",
            path=train_py,
            text=clip_lines(tma, 170),
        )
    )

                                
    snippets.append(
        Snippet(
            title="Training CLI entrypoint", path=train_script_py, text=clip_lines(read_text(train_script_py), 140)
        )
    )

                             
    e_txt = read_text(eval_py)
    emc = extract_function(e_txt, "evaluate_model_comprehensive") or ""
    base = extract_function(e_txt, "evaluate_baseline_naive") or ""
    snippets.append(
        Snippet(
            title="Evaluation metrics (including delta->price conversion)",
            path=eval_py,
            text=clip_lines((emc + "\n\n" + base).strip() or e_txt, 170),
        )
    )

                             
    snippets.append(
        Snippet(
            title="Batch runner (train + eval list of tickers)",
            path=pipeline_py,
            text=clip_lines(read_text(pipeline_py), 140),
        )
    )

    return snippets


def build_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

                   
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=14)
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.6,
        leading=10.2,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Stock Price Prediction – Project Explanation",
        author=os.getenv("USERNAME", ""),
    )

    story = []

    story.append(Paragraph("Stock Price Prediction – Project Explanation + Presentation Notes", h1))
    story.append(
        Paragraph(
            "This document explains the project end-to-end (data → features → model → training → evaluation) and includes key code excerpts from the repository. It also includes a short presentation script you can use when presenting to your professor.",
            body,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("1) What This Project Does", h2))
    story.append(
        Paragraph(
            "Goal: predict the next-day stock close (or next-day price change). The system downloads market data, builds features (OHLCV history window + VIX + sentiment + technical indicators), trains a deep learning model (BiLSTM + attention + residual MLP head), and evaluates with comprehensive plots and baseline comparisons.",
            body,
        )
    )

    story.append(Spacer(1, 8))
    story.append(Paragraph("2) End-to-End Data Flow (Big Picture)", h2))
    story.append(
        Paragraph(
            "1) scripts/train_advanced_model.py calls src.data_gathering.gather_data() to produce X (features) and y (targets).<br/>"
            "2) src.train.train_model_advanced() does a time-ordered split (train/val/test), fits scalers on train only, trains the model with early stopping, and saves the best checkpoint.<br/>"
            "3) scripts/evaluate_advanced_model.py reloads the saved model + scalers and evaluates on the last 15% of data, including a naive persistence baseline.",
            body,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("3) Key Design Decisions (What To Tell a Professor)", h2))
    story.append(
        Paragraph(
            "- Time-series split (no shuffling across train/val/test) to reduce leakage.<br/>"
            "- Train-only scaling: scalers are fit on training data and reused for validation/test.<br/>"
            "- Two target modes: price (next close) and delta (next-day change). Delta targets can be more stable.<br/>"
            "- Sentiment features are shifted by one day to reduce after-hours leakage.<br/>"
            "- Baseline is included: naive persistence (tomorrow ≈ today).",
            body,
        )
    )

    story.append(Spacer(1, 10))

    story.append(Paragraph("4) File-by-File Explanation", h2))
    story.append(
        Paragraph(
            "Below is what each file is responsible for, followed by a relevant code excerpt.",
            body,
        )
    )

    snippets = make_snippets()

    for snip in snippets:
        story.append(Spacer(1, 10))
        story.append(Paragraph(snip.title, ParagraphStyle("FileTitle", parent=h2, fontSize=12)))
        story.append(Paragraph(f"File: {snip.path.relative_to(ROOT).as_posix()}", body))

                                       
        explain = ""
        rel = snip.path.relative_to(ROOT).as_posix()
        if rel.endswith("src/config.py"):
            explain = (
                "Defines global constants like HISTORY_DAYS (how many days of OHLCV history per sample) and TARGET_MODE (price vs delta)."
            )
        elif rel.endswith("src/data_gathering.py"):
            explain = (
                "Builds the supervised dataset. For each day t, it concatenates an OHLCV window (t-history+1..t), sentiment, VIX, and technical indicators. Target is day t+1 price or delta."
            )
        elif rel.endswith("src/preprocessing.py"):
            explain = "Scales features and targets using StandardScaler. In training, the scaler is fit on training data only to avoid leakage."
        elif rel.endswith("src/dataset.py"):
            explain = "Minimal PyTorch Dataset wrapper that returns tensors for X and y with correct shapes."
        elif rel.endswith("src/model.py"):
            explain = (
                "Defines AdvancedStockPredictor. It reshapes the flattened OHLCV window into a sequence [batch, HISTORY_DAYS, 5], appends extra features across timesteps, runs BiLSTM, then self-attention, then residual MLP to output a single prediction."
            )
        elif rel.endswith("src/train.py"):
            explain = (
                "Implements the training loop: time-ordered split, scaling, DataLoaders, AdamW optimizer, LR scheduler, early stopping, gradient clipping, checkpoint saving, and test metrics."
            )
        elif rel.endswith("scripts/train_advanced_model.py"):
            explain = "CLI entrypoint for training a single ticker. It gathers data, calls train_model_advanced, and saves scalers + reports."
        elif rel.endswith("scripts/evaluate_advanced_model.py"):
            explain = "Evaluation pipeline: loads model+scalers, computes metrics (RMSE/MAE/MAPE/R², directional accuracy), compares against naive baseline, and generates plots."
        elif rel.endswith("scripts/run_pipeline.py"):
            explain = "Batch runner: trains/evaluates one ticker or a list from config/stocks_to_train.txt. Useful for coursework runs." 

        if explain:
            story.append(Paragraph(explain, body))

        story.append(Spacer(1, 6))
        story.append(Preformatted(snip.text, code_style))

    story.append(PageBreak())

    story.append(Paragraph("5) How To Present This (Slide Outline + Speaker Notes)", h2))

    story.append(Paragraph("Suggested slide outline (8–10 slides)", ParagraphStyle("B", parent=body, spaceAfter=6)))
    story.append(
        Paragraph(
            "1. Goal & problem framing (predict next-day close or delta).<br/>"
            "2. Data sources (Yahoo Finance OHLCV, VIX, optional NewsAPI).<br/>"
            "3. Feature engineering (OHLCV window, indicators, sentiment, VIX).<br/>"
            "4. Leakage prevention (time split, train-only scaling, shifted sentiment).<br/>"
            "5. Model architecture (BiLSTM + attention + residual head).<br/>"
            "6. Training setup (loss, optimizer, early stopping, checkpoints).<br/>"
            "7. Evaluation metrics + baseline (persistence).<br/>"
            "8. Results (show 1–2 tickers; emphasize baseline comparison).<br/>"
            "9. Limitations (price autocorrelation, direction near 50%, API reliability).<br/>"
            "10. Future work (multi-horizon returns, classification, regime modeling).",
            body,
        )
    )

    story.append(Spacer(1, 10))
    story.append(Paragraph("Speaker notes (short script)", ParagraphStyle("B2", parent=body, spaceAfter=6)))
    story.append(
        Paragraph(
            "\"This project builds a full pipeline for next-day stock prediction. I download OHLCV price data and market volatility (VIX), and optionally enrich it with NewsAPI headlines which are cached locally. For each day t, I build a feature vector consisting of a HISTORY_DAYS OHLCV window plus sentiment, VIX, and technical indicators. The target is either the next-day close price or the next-day delta.\"",
            body,
        )
    )
    story.append(
        Paragraph(
            "\"The model is a bidirectional LSTM over the OHLCV sequence with a self-attention layer to focus on important timesteps, followed by residual fully-connected blocks and a final regression head. Training uses a time-based split, scalers fit on training only, early stopping, gradient clipping, and checkpoints.\"",
            body,
        )
    )
    story.append(
        Paragraph(
            "\"For evaluation, I compute RMSE/MAE/MAPE/R² and directional accuracy, and I compare against a strong baseline: predicting tomorrow equals today. This baseline is hard to beat for next-day prices, so I always include it to avoid misleading results.\"",
            body,
        )
    )

    story.append(Spacer(1, 10))
    story.append(Paragraph("Common professor questions (and good answers)", ParagraphStyle("B3", parent=body, spaceAfter=6)))
    story.append(
        Paragraph(
            "Q: How did you prevent data leakage?<br/>"
            "A: Time-ordered split (shuffle=False), scalers fit on train only, sentiment shifted by 1 day, indicators computed from past only.<br/><br/>"
            "Q: Why do you report a naive baseline?<br/>"
            "A: Stock prices are highly autocorrelated. A model can look good on R² without adding value. Baseline shows whether the model truly improves.<br/><br/>"
            "Q: Why delta targets?<br/>"
            "A: Deltas are closer to stationary than prices and often avoid models collapsing to a mean price. In evaluation, deltas are converted back into price space for metrics."
            ,
            body,
        )
    )

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Generated from current workspace code. If you rerun training/evaluation, regenerate this PDF to keep snippets and explanations in sync.",
            ParagraphStyle("Foot", parent=body, fontSize=9, textColor="#444444"),
        )
    )

    doc.build(story)


def main() -> int:
    build_pdf(OUT_PDF)
    print(f"[OK] Wrote: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

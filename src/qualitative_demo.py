#!/usr/bin/env python3
"""
qualitative_demo.py

Generate a qualitative demo report for a model:
- Load a small dataset (CSV/JSONL)
- Run predictions
- Produce an HTML report with inputs, preds, and optionally labels
- Save a JSONL of predictions too

Usage:
  python qualitative_demo.py \
    --input data/qual_demo.jsonl \
    --outdir demos/run_001 \
    --task multiclass \
    --text_key text \
    --label_key label \
    --id_key id \
    --max_examples 100

Input format (JSONL or CSV) should contain at least:
  - id (optional)
  - text (or image_path) depending on your task
  - label (optional)

You must implement `predict_batch(examples)` for your project.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

TaskType = Literal["binary", "multiclass", "multilabel"]


# ---------------- IO ----------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    raise ValueError("Unsupported input type. Use .jsonl or .csv")


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Prediction stub ----------------
# Replace this with your project’s model loading and inference code.

def predict_batch(texts: Sequence[str], task: TaskType) -> np.ndarray:
    """
    Return probabilities for each example.

    - binary: shape (N,) with prob of class 1
    - multiclass: shape (N,C)
    - multilabel: shape (N,C) with independent probs

    TODO: Implement using your model.
    """
    raise NotImplementedError(
        "Implement predict_batch(texts, task) using your model, "
        "or wrap your existing inference code here."
    )


def probs_to_preds(
    probs: np.ndarray,
    task: TaskType,
    threshold: float,
) -> Any:
    if task == "binary":
        p = probs.reshape(-1)
        return (p >= threshold).astype(int), p
    if task == "multiclass":
        pred = np.argmax(probs, axis=-1).astype(int)
        score = np.max(probs, axis=-1)
        return pred, score
    if task == "multilabel":
        pred = (probs >= threshold).astype(int)
        return pred, probs
    raise ValueError(task)


# ---------------- Report ----------------

def render_html_report(
    rows: Sequence[Dict[str, Any]],
    outpath: Path,
    title: str,
    text_key: str,
    pred_key: str,
    score_key: str,
    label_key: Optional[str] = None,
    id_key: Optional[str] = None,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    def esc(x: Any) -> str:
        return html.escape("" if x is None else str(x))

    html_parts: List[str] = []
    html_parts.append(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(title)}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 12px; }}
small {{ color: #555; }}
.card {{ border: 1px solid #ddd; border-radius: 12px; padding: 14px; margin: 12px 0; }}
.row {{ display: grid; grid-template-columns: 140px 1fr; gap: 10px; }}
.k {{ color: #666; font-weight: 600; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #ccc; margin-right: 6px; }}
</style>
</head>
<body>
<h1>{esc(title)}</h1>
<small>Examples: {len(rows)}</small>
"""
    )

    for r in rows:
        rid = esc(r.get(id_key)) if id_key else ""
        text = esc(r.get(text_key, ""))
        pred = esc(r.get(pred_key, ""))
        score = esc(r.get(score_key, ""))
        label = esc(r.get(label_key, "")) if label_key and label_key in r else ""

        header_bits = []
        if rid:
            header_bits.append(f"<span class='tag'>id: {rid}</span>")
        if label_key and label:
            header_bits.append(f"<span class='tag'>label: {label}</span>")
        header_bits.append(f"<span class='tag'>pred: {pred}</span>")
        header_bits.append(f"<span class='tag'>score: {score}</span>")

        html_parts.append(
            f"""
<div class="card">
  <div>{''.join(header_bits)}</div>
  <div class="row" style="margin-top:10px;">
    <div class="k">input</div>
    <div><pre>{text}</pre></div>
  </div>
</div>
"""
        )

    html_parts.append("</body></html>")
    outpath.write_text("".join(html_parts), encoding="utf-8")
    print(f"[ok] wrote {outpath}")


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a qualitative demo HTML report.")
    p.add_argument("--input", required=True, help="Path to .jsonl or .csv containing examples.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--task", required=True, choices=["binary", "multiclass", "multilabel"])
    p.add_argument("--text_key", default="text", help="Key/column containing input text.")
    p.add_argument("--label_key", default="", help="Optional key/column containing ground-truth label.")
    p.add_argument("--id_key", default="id", help="Optional key/column containing example id.")
    p.add_argument("--max_examples", type=int, default=100, help="Max examples to include.")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary/multilabel.")
    p.add_argument("--title", default="Qualitative Demo", help="Report title.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(in_path)[: args.max_examples]

    texts = [str(r.get(args.text_key, "")) for r in rows]
    probs = predict_batch(texts, task=args.task)  # <- you implement this

    preds, scores = probs_to_preds(probs, task=args.task, threshold=args.threshold)

    # Attach outputs back to rows for exporting
    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["pred"] = (
            int(preds[i]) if np.ndim(preds) == 1 else preds[i].tolist()
        )
        rr["score"] = (
            float(scores[i]) if np.ndim(scores) == 1 else np.asarray(scores[i]).tolist()
        )
        out_rows.append(rr)

    # Save JSONL predictions
    write_jsonl(out_rows, out_dir / "preds.jsonl")

    # Write HTML report
    render_html_report(
        out_rows,
        outpath=out_dir / "report.html",
        title=args.title,
        text_key=args.text_key,
        pred_key="pred",
        score_key="score",
        label_key=args.label_key if args.label_key else None,
        id_key=args.id_key if args.id_key else None,
    )


if __name__ == "__main__":
    main()
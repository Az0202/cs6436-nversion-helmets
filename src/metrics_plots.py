#!/usr/bin/env python3
"""
metrics_plots.py

Load training/eval metrics from CSV or JSONL and generate line plots.

Examples:
  python metrics_plots.py --input runs/metrics.csv --outdir plots
  python metrics_plots.py --input runs/metrics.jsonl --outdir plots --x step --metrics loss,accuracy,val_loss
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def load_csv(path: Path) -> List[Dict[str, Union[str, float]]]:
    import csv

    rows: List[Dict[str, Union[str, float]]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, Union[str, float]] = {}
            for k, v in r.items():
                if v is None:
                    continue
                v = v.strip()
                if v == "":
                    continue
                row[k] = float(v) if _is_number(v) else v
            rows.append(row)
    return rows


def load_jsonl(path: Path) -> List[Dict[str, Union[str, float]]]:
    rows: List[Dict[str, Union[str, float]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Coerce numeric strings to floats
            row: Dict[str, Union[str, float]] = {}
            for k, v in obj.items():
                if isinstance(v, str) and _is_number(v):
                    row[k] = float(v)
                else:
                    row[k] = v
            rows.append(row)
    return rows


def load_metrics(path: Path) -> List[Dict[str, Union[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix in (".jsonl", ".jl"):
        return load_jsonl(path)
    raise ValueError(f"Unsupported input type: {suffix}. Use .csv or .jsonl")


def available_numeric_keys(rows: Sequence[Dict[str, Union[str, float]]]) -> List[str]:
    keys = set()
    for r in rows:
        keys.update(r.keys())
    numeric_keys = []
    for k in sorted(keys):
        # consider numeric if any row has float/int under k
        if any(isinstance(r.get(k), (int, float)) for r in rows):
            numeric_keys.append(k)
    return numeric_keys


def extract_series(
    rows: Sequence[Dict[str, Union[str, float]]],
    x_key: str,
    y_key: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        x = r.get(x_key)
        y = r.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            xs.append(float(x))
            ys.append(float(y))
    # If x is missing but y exists, fall back to index
    if not xs and any(isinstance(r.get(y_key), (int, float)) for r in rows):
        for i, r in enumerate(rows):
            y = r.get(y_key)
            if isinstance(y, (int, float)):
                xs.append(float(i))
                ys.append(float(y))
    return xs, ys


def plot_metric(
    rows: Sequence[Dict[str, Union[str, float]]],
    x_key: str,
    metric: str,
    outpath: Path,
    title: Optional[str] = None,
) -> None:
    xs, ys = extract_series(rows, x_key=x_key, y_key=metric)
    if len(xs) < 2:
        print(f"[skip] Not enough points for '{metric}' (found {len(xs)})")
        return

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(x_key if xs else "index")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[ok] wrote {outpath}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot metrics from CSV or JSONL.")
    p.add_argument("--input", required=True, help="Path to metrics.csv or metrics.jsonl")
    p.add_argument("--outdir", required=True, help="Output directory for PNG plots")
    p.add_argument(
        "--x",
        default="step",
        help="X-axis key/column (default: step). If missing, index is used.",
    )
    p.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metric keys to plot. If empty, plots all numeric keys except x.",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix for plot images.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.outdir)

    rows = load_metrics(in_path)
    if not rows:
        raise RuntimeError(f"No rows found in {in_path}")

    numeric = available_numeric_keys(rows)

    x_key = args.x
    metrics: List[str]
    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = [k for k in numeric if k != x_key]

    if not metrics:
        raise RuntimeError(
            f"No metrics to plot. Numeric keys found: {numeric}. "
            f"Try --metrics or adjust --x."
        )

    # Warn if x_key not present / not numeric
    if x_key not in numeric:
        print(
            f"[warn] x key '{x_key}' not found as numeric. "
            f"Will fall back to index for metrics that exist."
        )

    for m in metrics:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in m)
        fname = f"{args.prefix}{safe}.png" if args.prefix else f"{safe}.png"
        plot_metric(
            rows,
            x_key=x_key,
            metric=m,
            outpath=out_dir / fname,
            title=f"{m} vs {x_key}",
        )


if __name__ == "__main__":
    main()
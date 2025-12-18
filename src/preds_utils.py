"""
preds_utils.py

Utilities for working with model predictions ("preds"):
- Convert logits -> probabilities (sigmoid/softmax)
- Binary/multiclass thresholding + top-k
- Decode class IDs to labels
- Simple ensembling
- Save/load predictions (jsonl/csv)

Designed to be framework-agnostic (works with lists / numpy arrays).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

TaskType = Literal["binary", "multiclass", "multilabel"]


# ---------- basic transforms ----------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)  # stability
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def logits_to_probs(
    logits: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    task: TaskType,
) -> np.ndarray:
    """
    Convert logits to probabilities for common task types.

    - binary: returns shape (N,) or (N,1) -> (N,)
    - multiclass: softmax over last axis, returns shape (N,C)
    - multilabel: sigmoid per class, returns shape (N,C)
    """
    arr = np.asarray(logits, dtype=np.float64)
    if task == "binary":
        arr = arr.reshape(-1)
        return sigmoid(arr)
    if task == "multiclass":
        if arr.ndim == 1:
            # single example -> (C,)
            return softmax(arr, axis=-1)
        return softmax(arr, axis=-1)
    if task == "multilabel":
        if arr.ndim == 1:
            return sigmoid(arr)
        return sigmoid(arr)
    raise ValueError(f"Unknown task: {task}")


# ---------- decision rules ----------

def predict_binary(
    probs: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    return (p >= threshold).astype(np.int64)


def predict_multiclass(
    probs: Union[np.ndarray, Sequence[Sequence[float]]],
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        return np.array(int(np.argmax(p)), dtype=np.int64)
    return np.argmax(p, axis=-1).astype(np.int64)


def topk(
    probs: Union[np.ndarray, Sequence[Sequence[float]]],
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (topk_indices, topk_scores).
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        idx = np.argsort(-p)[:k]
        return idx.astype(np.int64), p[idx]
    idx = np.argsort(-p, axis=-1)[:, :k]
    scores = np.take_along_axis(p, idx, axis=-1)
    return idx.astype(np.int64), scores


def predict_multilabel(
    probs: Union[np.ndarray, Sequence[Sequence[float]]],
    threshold: float = 0.5,
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    return (p >= threshold).astype(np.int64)


def decode_labels(
    class_ids: Union[int, Sequence[int], np.ndarray],
    id_to_label: Union[Dict[int, str], Sequence[str]],
) -> Union[str, List[str]]:
    """
    Decode predicted class ids to human-readable labels.
    """
    if isinstance(id_to_label, dict):
        mapper = id_to_label
        lookup = lambda i: mapper.get(int(i), str(int(i)))
    else:
        labels = list(id_to_label)
        lookup = lambda i: labels[int(i)] if 0 <= int(i) < len(labels) else str(int(i))

    if isinstance(class_ids, (int, np.integer)):
        return lookup(class_ids)
    arr = np.asarray(class_ids)
    return [lookup(i) for i in arr.reshape(-1).tolist()]


# ---------- ensembling ----------

def ensemble_mean(
    probs_list: Sequence[Union[np.ndarray, Sequence]],
) -> np.ndarray:
    """
    Average probabilities from multiple models/runs.
    """
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in probs_list], axis=0)
    return np.mean(stacked, axis=0)


def ensemble_weighted(
    probs_list: Sequence[Union[np.ndarray, Sequence]],
    weights: Sequence[float],
) -> np.ndarray:
    """
    Weighted average of probabilities.
    """
    if len(probs_list) != len(weights):
        raise ValueError("probs_list and weights must have same length")
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in probs_list], axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


# ---------- IO helpers ----------

@dataclass
class PredRecord:
    """
    A common portable prediction record.

    id: sample id / key
    pred: predicted class id (multiclass) or 0/1 (binary) or list (multilabel)
    score: probability/confidence (float) or list of scores
    extra: any extra metadata you want to keep
    """
    id: str
    pred: Any
    score: Any
    extra: Optional[Dict[str, Any]] = None


def save_jsonl(preds: Sequence[PredRecord], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in preds:
            obj = {"id": r.id, "pred": r.pred, "score": r.score}
            if r.extra:
                obj.update(r.extra)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_csv(
    rows: Sequence[Dict[str, Any]],
    path: Union[str, Path],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- convenience: end-to-end helpers ----------

def make_pred_records_multiclass(
    ids: Sequence[str],
    probs: Union[np.ndarray, Sequence[Sequence[float]]],
    id_to_label: Optional[Union[Dict[int, str], Sequence[str]]] = None,
    k: int = 1,
) -> List[PredRecord]:
    """
    Create PredRecord list for multiclass probs.
    If k>1, stores top-k ids and scores.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)

    if k <= 1:
        pred_ids = np.argmax(p, axis=-1)
        scores = np.max(p, axis=-1)
        out: List[PredRecord] = []
        for i, pid, sc in zip(ids, pred_ids, scores):
            pred_val: Any = int(pid)
            if id_to_label is not None:
                pred_val = decode_labels(int(pid), id_to_label)
            out.append(PredRecord(id=str(i), pred=pred_val, score=float(sc)))
        return out

    idx, scs = topk(p, k=k)
    out = []
    for i, top_ids, top_scores in zip(ids, idx, scs):
        pred_vals: Any = [int(x) for x in top_ids.tolist()]
        if id_to_label is not None:
            pred_vals = decode_labels(pred_vals, id_to_label)
        out.append(
            PredRecord(
                id=str(i),
                pred=pred_vals,
                score=[float(x) for x in top_scores.tolist()],
            )
        )
    return out
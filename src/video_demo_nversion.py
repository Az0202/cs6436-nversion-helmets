#!/usr/bin/env python3
"""
video_demo_nversion.py

Compare N model versions on the same video(s) and render an annotated demo video.

- Loads one input video (or a folder of videos)
- Loads multiple model versions/checkpoints
- Runs inference per frame (optionally every Nth frame)
- Overlays predictions from each version on the frame
- Writes an output video + a JSON summary

Usage:
  python video_demo_nversion.py \
    --video input.mp4 \
    --versions v1=ckpts/model_v1.pt v2=ckpts/model_v2.pt v3=ckpts/model_v3.pt \
    --outdir demos/compare_001 \
    --fps_out 30 \
    --stride 1 \
    --max_frames 1500

If you want multiple videos:
  --video_dir videos/ --glob "*.mp4"

Dependencies:
  pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("Missing dependency: opencv-python. Install via `pip install opencv-python`.") from e


# ------------------ Project hooks (YOU implement) ------------------

@dataclass
class ModelHandle:
    name: str
    ckpt_path: Path
    obj: Any  # your model object


def load_model(name: str, ckpt_path: Path) -> ModelHandle:
    """
    Load and return a model handle for a given checkpoint.

    TODO: Replace this with your framework-specific loading:
      - PyTorch: torch.load / model.load_state_dict
      - TF: tf.saved_model.load
      - ONNX: onnxruntime.InferenceSession
    """
    raise NotImplementedError("Implement load_model(name, ckpt_path) for your project.")


def predict_frame(model: ModelHandle, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Run prediction for a single frame.

    Return a dict that is easy to display, e.g.:
      {"label": "cat", "score": 0.93}
      or {"boxes": [...], "scores": [...], "labels": [...]}

    TODO: Implement using your model.
    """
    raise NotImplementedError("Implement predict_frame(model, frame_bgr) for your project.")


def format_pred_for_overlay(pred: Dict[str, Any]) -> str:
    """
    Convert prediction dict to a short single-line string for overlay.
    Customize based on your task.
    """
    if "label" in pred:
        sc = pred.get("score")
        return f"{pred['label']} ({sc:.2f})" if isinstance(sc, (int, float)) else str(pred["label"])
    if "text" in pred:
        return str(pred["text"])
    # fallback
    return json.dumps(pred, ensure_ascii=False)[:120]


# ------------------ Utility: parsing versions ------------------

def parse_versions(items: Sequence[str]) -> List[Tuple[str, Path]]:
    """
    Parse CLI args like: ["v1=path1", "v2=path2"]
    """
    out: List[Tuple[str, Path]] = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --versions entry: '{it}'. Use name=path.")
        name, p = it.split("=", 1)
        name = name.strip()
        pth = Path(p).expanduser()
        if not name:
            raise ValueError(f"Empty version name in '{it}'")
        out.append((name, pth))
    return out


# ------------------ Overlay helpers ------------------

def draw_overlay(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (12, 28),
    line_h: int = 28,
) -> np.ndarray:
    """
    Draw overlay text lines onto a frame.
    Note: we don't set colors explicitly (OpenAI tool style guidance doesn't apply here,
    but we keep it simple anyway).
    """
    x0, y0 = origin
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    for i, t in enumerate(lines):
        y = y0 + i * line_h
        # shadow
        cv2.putText(out, t, (x0 + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # text
        cv2.putText(out, t, (x0, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# ------------------ Core runner ------------------

def process_video(
    video_path: Path,
    models: List[ModelHandle],
    out_path: Path,
    fps_out: Optional[float],
    stride: int,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(fps_out) if fps_out else float(fps_in)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    stats: Dict[str, Any] = {"video": str(video_path), "fps_in": fps_in, "fps_out": fps, "frames": 0, "versions": {}}
    for m in models:
        stats["versions"][m.name] = {"ckpt": str(m.ckpt_path), "pred_samples": []}

    frame_idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # Run each model on this frame
        lines = [f"video: {video_path.name} | frame: {frame_idx}"]
        for m in models:
            pred = predict_frame(m, frame)
            s = f"{m.name}: {format_pred_for_overlay(pred)}"
            lines.append(s)

            # store a few samples for debugging
            if len(stats["versions"][m.name]["pred_samples"]) < 25:
                stats["versions"][m.name]["pred_samples"].append({"frame": frame_idx, "pred": pred})

        annotated = draw_overlay(frame, lines)
        writer.write(annotated)

        written += 1
        frame_idx += 1
        if max_frames is not None and written >= max_frames:
            break

    cap.release()
    writer.release()
    stats["frames"] = written
    return stats


def list_videos(video_dir: Path, glob_pat: str) -> List[Path]:
    return sorted(video_dir.glob(glob_pat))


# ------------------ CLI ------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an N-version video demo comparison.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="Input video path.")
    g.add_argument("--video_dir", type=str, help="Directory of videos.")
    p.add_argument("--glob", type=str, default="*.mp4", help="Glob pattern for --video_dir (default: *.mp4).")

    p.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help="List of name=checkpoint_path entries (e.g., v1=ckpt1.pt v2=ckpt2.pt).",
    )

    p.add_argument("--outdir", type=str, required=True, help="Output directory.")
    p.add_argument("--fps_out", type=float, default=None, help="Override output fps (default: input fps).")
    p.add_argument("--stride", type=int, default=1, help="Run inference every Nth frame (default: 1).")
    p.add_argument("--max_frames", type=int, default=None, help="Limit number of processed frames.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions = parse_versions(args.versions)
    models: List[ModelHandle] = []
    for name, ckpt in versions:
        models.append(load_model(name, ckpt))

    videos: List[Path]
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = list_videos(Path(args.video_dir), args.glob)

    all_stats: List[Dict[str, Any]] = []
    for vp in videos:
        out_path = out_dir / f"{vp.stem}__compare.mp4"
        stats = process_video(
            video_path=vp,
            models=models,
            out_path=out_path,
            fps_out=args.fps_out,
            stride=max(1, args.stride),
            max_frames=args.max_frames,
        )
        all_stats.append(stats)
        print(f"[ok] wrote {out_path} ({stats['frames']} frames)")

    (out_dir / "summary.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
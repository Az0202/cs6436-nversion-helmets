# N-Version Model Evaluation & Visualization

<p align="center">
  <img src="assets/ReadMe_icon.png" alt="N-Version Head/Helmet Detection for PPE Safety on Construction Sites" width="600">
</p>

This repository contains utilities and demos for evaluating and comparing
multiple object-detection model outputs (YOLOv8, Faster R-CNN, and ensembles)
for a 2-class hardhat detection task.

The code is intended for **coursework / research use**, focusing on:
- metric comparison across model versions
- prediction post-processing
- qualitative visualization
- N-version video comparison

This is **not** a production framework. Scripts are designed to be run
programmatically or from notebooks.

---

## Repository Structure

```
.
├── data/
│   ├── data_hardhat_2class.yaml
│   ├── metrics_map_summary_nversion.csv
│   ├── preds_frcnn_seed0_val.json
│   ├── preds_yolov8n_seed0_val.json
│   ├── preds_yolov8s_seed0_val.json
│   ├── preds_wbf_yolov8n_yolov8s_val.json
│   ├── preds_wbf_yolov8n_yolov8s_frcnn_val.json
│   └── preds_majority_yolov8n_yolov8s_frcnn_val.json
│
├── notebooks/
│   ├── CS_6433_project_code1.ipynb
│   └── demo_CS_6434.ipynb
│
├── src/
│   ├── paths_config.py
│   ├── metrics_plots.py
│   ├── preds_utils.py
│   ├── qualitative_demo.py
│   └── video_demo_nversion.py
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Overview

### Dataset

This project uses the **Hard Hat Detection** dataset from Kaggle:

🔗 **[Hard Hat Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)**

The dataset contains images of construction workers with annotations for:
- `helmet` - workers wearing hardhats
- `head` - workers without hardhats

> **Note:** The images are not included in this repository. Download the dataset from Kaggle to run the full pipeline.

### Dataset configuration

- `data/data_hardhat_2class.yaml`

YAML configuration for the 2-class hardhat detection dataset.

### Prediction files

- `preds_*.json`

Validation predictions from:
- YOLOv8n
- YOLOv8s
- Faster R-CNN
- Weighted Box Fusion (WBF)
- Majority-vote ensemble

These files are included intentionally for reproducibility and analysis.

### Metrics summary

- `metrics_map_summary_nversion.csv`

Aggregated mAP results across individual models and ensemble methods.

---

## Path Configuration

`src/paths_config.py` defines fixed project paths using module-level variables.

Example:

```python
from paths_config import DATA_DIR, OUTPUT_DIR

print(DATA_DIR)
print(OUTPUT_DIR)
```

Paths are not configurable via environment variables in the current
implementation.

---

## Metrics Plotting

`metrics_plots.py` generates plots from the metrics summary CSV.

### How it works
- Paths are imported from `paths_config.py`
- No CLI arguments are used
- When executed, the script automatically generates all plots

Run:

```bash
python src/metrics_plots.py
```

Plots are saved to the output directory defined in `paths_config.py`.

---

## Prediction Utilities

`preds_utils.py` provides helper functions for working with object-detection
predictions, including:
- loading and saving prediction JSON files
- bounding box utilities
- IoU computation
- non-maximum suppression (NMS)
- box format conversions

Example usage:

```python
from preds_utils import load_predictions, compute_iou

preds = load_predictions("data/preds_yolov8n_seed0_val.json")
iou = compute_iou(box1, box2)
```

This module does **not** implement:
- logits-to-probability conversion
- classification utilities
- ensembling logic (ensembles are precomputed in the data folder)

---

## Qualitative Visualization

`qualitative_demo.py` generates 6-panel image visualizations comparing
predictions from multiple model versions.

### Key characteristics
- No CLI interface
- Intended to be run from a notebook or imported as a module
- Uses `create_6panel_visualization()` to render side-by-side comparisons

Typical usage (from a notebook):

```python
from qualitative_demo import create_6panel_visualization, load_predictions

# Load predictions into a dictionary
predictions_dict = {
    "yolov8n": load_predictions("data/preds_yolov8n_seed0_val.json"),
    "yolov8s": load_predictions("data/preds_yolov8s_seed0_val.json"),
    "frcnn": load_predictions("data/preds_frcnn_seed0_val.json"),
    "wbf_2": load_predictions("data/preds_wbf_yolov8n_yolov8s_val.json"),
    "wbf_3": load_predictions("data/preds_wbf_yolov8n_yolov8s_frcnn_val.json"),
    "majority": load_predictions("data/preds_majority_yolov8n_yolov8s_frcnn_val.json"),
}

create_6panel_visualization(
    image_path="path/to/image.jpg",
    image_id="image_001",
    predictions_dict=predictions_dict,
    conf_threshold=0.5,
    save_path="outputs/figures/comparison.jpg"
)
```

---

## N-Version Video Demo

`video_demo_nversion.py` performs video-based comparison across multiple
detectors.

### Features
- Uses a built-in `YOLODetector` class
- Processes videos frame-by-frame
- Overlays predictions from multiple model versions
- Designed to be called programmatically

Example usage:

```python
from video_demo_nversion import process_video, YOLODetector

# Initialize detectors
models = [
    YOLODetector("checkpoints/yolov8n.pt"),
    YOLODetector("checkpoints/yolov8s.pt"),
]

# Process video
process_video(
    input_path="input.mp4",
    output_path="output_comparison.mp4",
    models=models,
    display=False
)
```

This script:
- Does **not** use CLI arguments
- Does **not** output a summary JSON
- Is intended for controlled experiments, not batch pipelines

---

## Notebooks

The `notebooks/` directory contains the main analysis and demo notebooks:

| Notebook | Description |
|----------|-------------|
| `CS_6433_project_code1.ipynb` | Main project code - model training, evaluation, and metrics |
| `demo_CS_6434.ipynb` | Video demonstration and qualitative analysis |

### 🚀 Quick Start with Google Colab

**Don't want to clone the repo and set up locally?** You can run the notebooks directly in Google Colab:

1. Download the notebook you want:
   - [`CS_6433_project_code1.ipynb`](notebooks/CS_6433_project_code1.ipynb)
   - [`demo_CS_6434.ipynb`](notebooks/demo_CS_6434.ipynb)

2. Go to [Google Colab](https://colab.research.google.com/)

3. Click **File → Upload notebook** and select the downloaded file

4. Run the cells!

> **Note:** Some cells may require uploading data files or adjusting paths for the Colab environment.

---

## Notes

- Prediction JSON files are included intentionally.
- Scripts are tightly coupled to the current project structure.
- This repository prioritizes clarity and reproducibility over flexibility.

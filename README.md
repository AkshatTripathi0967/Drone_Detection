# 🚁 Drone Detection

An object detection project that identifies drones in images and video using deep learning. The model is trained on the Kaggle Drone Detection dataset and evaluated with standard detection metrics (Precision, Recall, mAP).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🔍 Overview

Drones are small, fast-moving and often appear against cluttered backgrounds, which makes them hard to detect. This project trains an object detector to locate drones and draw bounding boxes around them, with a confidence score for each detection.

**Key features**
- Detects drones in images, videos and live camera streams
- Reports per-detection confidence scores
- Evaluated with Precision, Recall, mAP@50 and mAP@50–95

---

## 📂 Dataset

The model is trained on the **Drone Detection** dataset from Kaggle:

🔗 **[Drone Detection Dataset on Kaggle](https://www.kaggle.com/datasets/meghrajdesai/drone-detection)**

**Downloading the dataset**

```bash
pip install kaggle
kaggle datasets download -d meghrajdesai/drone-detection
unzip drone-detection.zip -d data/
```

> You need a Kaggle account and API token (`kaggle.json`). See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

Please review the dataset page for its license and terms of use before using it.

---

## 📊 Results

Performance on the validation set:

| Metric | Score | Remarks |
|--------|:-----:|---------|
| **Precision** | **94.2%** | Excellent |
| **Recall** | **67.5%** | Room to improve |
| **mAP@50** | **78.0%** | Good |
| **mAP@50–95** | **48.0%** | Moderate |

### What these numbers mean

- **Precision (94.2%):** When the model says "drone", it is right about 94 times out of 100. False alarms are rare.
- **Recall (67.5%):** The model finds roughly two-thirds of the drones in the data, so about one in three is missed.
- **mAP@50 (78.0%):** Average precision across recall levels when a box counts as correct if it overlaps the ground truth by at least 50% (IoU ≥ 0.5).
- **mAP@50–95 (48.0%):** The same measure averaged over stricter IoU thresholds (0.50 to 0.95). The gap from mAP@50 suggests detections are found but bounding boxes are not always tightly aligned.

### Analysis

The model is **conservative**: very precise, but it misses a noticeable share of drones. Likely causes are very small or distant drones, motion blur, low contrast against the sky or background, and partial occlusion. The recall gap is the main area to improve (see [Future Improvements](#-future-improvements)).

---

## 🗂️ Project Structure

> Update this section to match your repository.

```
drone-detection/
├── data/                  # Dataset (not tracked by git)
├── models/                # Trained weights
├── notebooks/             # Training / experiments
├── src/
│   ├── train.py           # Training script
│   ├── detect.py          # Inference script
│   └── utils.py           # Helper functions
├── results/               # Metrics, plots, sample predictions
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/drone-detection.git
cd drone-detection
```

**2. Create a virtual environment**

```bash
python -m venv drone-env

# Windows
drone-env\Scripts\activate
# Mac / Linux
source drone-env/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

> Replace the commands below with the ones that match your scripts.

**Detect drones in an image**

```bash
python src/detect.py --source path/to/image.jpg --weights models/best.pt
```

**Detect drones in a video**

```bash
python src/detect.py --source path/to/video.mp4 --weights models/best.pt
```

**Use a webcam**

```bash
python src/detect.py --source 0 --weights models/best.pt
```

Output images and videos with bounding boxes are saved to the `results/` folder.

---

## 🏋️ Training

```bash
python src/train.py --data data/drone.yaml --epochs 100 --img 640 --batch 16
```

| Setting | Value |
|---------|-------|
| Model | *[e.g. YOLOv8n / YOLOv8s]* |
| Image size | *[e.g. 640×640]* |
| Epochs | *[e.g. 100]* |
| Batch size | *[e.g. 16]* |
| Optimizer | *[e.g. AdamW]* |

---

## 🔮 Future Improvements

Ways to raise recall and mAP@50–95:

- **Lower the confidence threshold** at inference to trade some precision for more detections
- **Higher input resolution** (e.g. 960 or 1280) to help with small, distant drones
- **More data augmentation** such as mosaic, scale, blur and brightness changes
- **Add more varied training data** (different weather, lighting, backgrounds and drone types)
- **Use a larger model** or try a different architecture
- **Tile / slice inference** for high-resolution images with tiny drones
- **Hard-negative mining** and review of missed detections to find failure patterns

---

## 🙏 Acknowledgements

- Dataset: [Drone Detection by Megh Raj Desai](https://www.kaggle.com/datasets/meghrajdesai/drone-detection) on Kaggle
- Built with *[Python, PyTorch / TensorFlow, Ultralytics, OpenCV, etc.]*

---

## 📄 License

This project is released under the **[MIT / Apache 2.0 / other]** license. See the `LICENSE` file for details. The dataset is subject to its own license on Kaggle.

---

## 📬 Contact

**[Akshat Tripathi]**: [akshatt030@gmail.com] · [GitHub] https://github.com/AkshatTripathi0967 · [LinkedIn] https://www.linkedin.com/in/akshat-tripathi-76122a2a2

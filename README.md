# Skin Lesion Classification using Deep Learning
This repository contains a deep learning pipeline designed to classify skin lesion dermoscopy images into one of seven diagnostic categories. Built using **PyTorch**, the project tracks experiments via **Weights & Biases (W&B)** and explores techniques ranging from custom Convolutional Neural Networks (CNNs) to transfer learning and data augmentation.

## 2. Dataset Overview
The dataset is a subset of the **ISIC 2018 Challenge (Task 3)** containing approximately 3,800 clinical images. The target diagnostic classes are:
* `MEL`: Melanoma
* `NV`: Melanocytic nevus
* `BCC`: Basal cell carcinoma
* `AKIEC`: Actinic keratosis
* `BKL`: Benign keratosis
* `DF`: Dermatofibroma
* `VASC`: Vascular lesion

---

## Repository Architecture

lesion-lens/
│
├── data/
│   ├── img/                  # Directory containing ~3,800 ISIC .jpg images
│   └── txt/                  # Manifest and label files
│       ├── classes.txt
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── train_small.csv   # Debugging subset (200 rows)
│       └── val_small.csv     # Debugging subset (200 rows)
│
├── saved_models/             # Generated model checkpoints (.pt / .pth)
├── logs/                     # Local training logs
│
├── explore.py                # Data analysis and visualization utilities
├── datasets.py               # Custom PyTorch Dataset class (LesionDataset)
├── models.py                 # Network architectures (SimpleBNConv, ResNet wrapper)
├── train.py                  # Training pipeline, validation loop, and metrics
│
├── notebook.ipynb            # Core execution notebook (Google Colab wrapper)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

---

## Technical Features & Pipeline

### 1. Data Exploration & Imbalance Mitigation
* Analysis of data splits using argmax conversions from one-hot labels.
* Assessment of multi-class distribution skewness to calculate corrective class weights or loss functions.

### 2. Baseline Architecture (`SimpleBNConv`)
* **5 Convolutional Layers** with incremental channel depths: 8, 16, 32, 64, and 128.
* Interleaved with `nn.ReLU()` activations and `nn.BatchNorm2d`.
* Progressive downsampling using `nn.MaxPool2d` (factor of 2).

### 3. Data Augmentation Engine
* Deterministic and non-deterministic image transformations configured exclusively for the training partition.
* Features random horizontal flips and extra spatial/color jitter adjustments to improve generalization.

### 4. Transfer Learning Configurations
* **Feature Extraction:** Freezing all pretrained layers (e.g., ResNet18) except the final linear classification layer.
* **Fine-Tuning:** Unfreezing the network to optimize all pre-trained weights globally.

---

## Setup & Installation

### Dependencies
Ensure you have Python 3.10+ installed. Install project requirements using:
```bash
pip install -r requirements.txt
# Lesion Lens: Deep Learning for Skin Cancer Classification

![Status](https://img.shields.io/badge/Status-Completed-success)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![MLOps](https://img.shields.io/badge/MLOps-Weights&Biases-yellow)

**Lesion Lens** is a high-accuracy, hybrid diagnostic pipeline for skin cancer classification. It combines the feature-extraction power of a deep convolutional neural network (CNN) with the clinical context provided by patient metadata, achieving 92% classification accuracy on the HAM10000 dataset.

---

## Key Features

*   **Deep Learning Backbone:** Implemented a custom ResNet50-based architecture for high-level image feature extraction.
*   **Clinical Hybridization:** Fused deep latent representations with engineered clinical features (age, sex, localization) using a *scikit-learn* classifier.
*   **Clinical Robustness:** Specifically engineered to handle severe class imbalances typical in medical datasets using `WeightedRandomSampler` and advanced data augmentation.
*   **Reproducible MLOps:** Integrated with **Weights & Biases** to track hyperparameter tuning and model performance metrics.

---

## Tech Stack

*   **Core:** Python, PyTorch, torchvision
*   **Classical ML:** scikit-learn (Random Forest integration)
*   **Data Handling:** Pandas, NumPy, PIL
*   **Monitoring:** Weights & Biases (W&B)

---

## Performance

| Approach | Classification Accuracy |
| :--- | :--- |
| CNN Only | 88% |
| **Hybrid (CNN + Clinical Features)** | **92%** |

---

## Project Structure

```text
├── data/               # Metadata and subset organization
├── models/             # Saved model checkpoints
├── notebooks/          # Exploratory analysis and visualizations
├── utils/              # Data loading and augmentation helpers
├── train.py            # Main training loop (PyTorch + W&B)
├── model.py            # CNN architecture definition
├── hybrid_inference.py # Feature extraction and fusion pipeline
└── requirements.txt    # Project dependencies
```
---

## How to Run

1. **Clone the repository:**
```bash
git clone [https://github.com/3m-6h7/lesion-lens.git](https://github.com/3m-6h7/lesion-lens.git)
   cd lesion-lens
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the training pipeline:**
```bash
python train.py
```

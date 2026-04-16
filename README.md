# Supervised vs. Unsupervised Pattern Recognition for SIEM False Positive Reduction

Companion code for the paper submitted to **Pattern Recognition** (Elsevier).

> Sohal, S. (2025). Supervised and Unsupervised Pattern Recognition for SIEM Alert Reduction. *Pattern Recognition* (under review, PR-D-25-05676).

---

## Overview

This repository implements a machine learning pipeline that compares **supervised** and **unsupervised** pattern recognition approaches for reducing false positive alerts in hybrid cloud Security Information and Event Management (SIEM) environments.

The problem is framed as a pattern recognition challenge: distinguishing normal network behaviour patterns from attack signatures under class imbalance, high dimensionality, and temporal concept drift.

**Dataset:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity

---

## Models Evaluated

| Paradigm | Algorithm |
|---|---|
| Supervised (discriminative) | Random Forest |
| Supervised (discriminative) | XGBoost |
| Unsupervised (anomaly detection) | Isolation Forest |
| Unsupervised (reconstruction-based) | Autoencoder |

---

## Key Results

| Model | Accuracy | FPR | FNR | ROC-AUC | Train (s) |
|---|---|---|---|---|---|
| Random Forest | 99.86 % | **0.12 %** | 0.28 % | 0.9999 | 1,081 |
| XGBoost | 99.78 % | 0.21 % | 0.24 % | 0.9999 | 129 |
| Isolation Forest | 83.91 % | 6.48 % | 70.27 % | 0.8367 | 19 |
| Autoencoder | 90.69 % | 1.87 % | 51.25 % | 0.9121 | 80 |

FPR (False Positive Rate) is the primary operational metric — it directly measures analyst workload in SIEM environments.

---

## Repository Structure

```
.
├── supervised_vs_unsupervised.ipynb   # Jupyter notebook (Colab-ready, 20 cells)
├── supervised_vs_unsupervised.py      # Standalone Python script
├── requirements.txt                   # Pinned dependencies
└── README.md
```

---

## Quick Start

### Option 1 — Google Colab (recommended)

Open `supervised_vs_unsupervised.ipynb` in Colab. All dependencies are installed automatically in Cell 1.

### Option 2 — Local Python

```bash
pip install -r requirements.txt
python supervised_vs_unsupervised.py          # full run (includes Autoencoder)
python supervised_vs_unsupervised.py --no-ae  # skip Autoencoder
```

Missing packages are installed automatically on first run.

---

## Pipeline

1. **Auto-install** — installs tensorflow, shap, xgboost, imbalanced-learn, etc. if missing
2. **Data loading** — downloads and merges all CIC-IDS2017 CSV files
3. **Preprocessing** — cleaning, feature selection, MinMaxScaler, RandomUnderSampler
4. **Training**
   - **Random Forest**: `GridSearchCV` (3-fold CV, weighted F1); grid over `n_estimators`, `max_depth`, `min_samples_split`
   - **XGBoost**: `GridSearchCV` (3-fold CV, weighted F1); grid over `n_estimators`, `learning_rate`, `max_depth`
   - **Isolation Forest**: fixed hyperparameters (`contamination=0.1`, `n_estimators=100`); trained on full unsampled training set — genuinely unsupervised
   - **Autoencoder**: trained on benign-only traffic; anomaly threshold = μ + 3σ of reconstruction error on benign training subset
5. **Evaluation** — Accuracy, Precision, Recall, F1, FPR, FNR, AUC, timing
6. **Visualisation** — ROC curves, Precision-Recall curves, Confusion matrices, FPR/FNR bar chart
7. **SHAP** — feature importance for supervised models

---

## Dependencies

See `requirements.txt`. Core packages:

- scikit-learn ≥ 1.2
- xgboost ≥ 1.7
- tensorflow ≥ 2.10 (optional, for Autoencoder)
- shap ≥ 0.42
- imbalanced-learn ≥ 0.10

---

## Data Availability

The CIC-IDS2017 dataset is publicly available from the Canadian Institute for Cybersecurity:
<https://www.unb.ca/cic/datasets/ids-2017.html>

---

## Citation

If you use this code, please cite the companion paper (citation to be updated upon acceptance).

# Supervised vs. Unsupervised Pattern Recognition for SIEM False Positive Reduction

Companion code for the paper submitted to **Pattern Recognition** (Elsevier).

> Sohal, S. (2025). Supervised vs. Unsupervised Pattern Recognition for SIEM False Positive Reduction. *Pattern Recognition* (under review).

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

| Model | Accuracy | FPR | FNR | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 99.66 % | **0.23 %** | 0.85 % | 0.9997 |
| XGBoost | 95.30 % | 4.74 % | 4.47 % | 0.9932 |
| Isolation Forest | 77.99 % | 8.53 % | 88.37 % | 0.6441 |
| Autoencoder | 88.83 % | 5.01 % | 41.44 % | 0.9060 |

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
   - Supervised: `GridSearchCV` (3-fold CV, weighted F1 objective)
   - Isolation Forest: grid search over contamination / n_estimators
   - Autoencoder: trained on benign-only traffic; threshold = mean + 3σ reconstruction error
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

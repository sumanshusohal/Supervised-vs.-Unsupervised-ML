#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised vs. Unsupervised Pattern Recognition for SIEM False Positive Reduction
==================================================================================
Companion code for:
  Sohal, S. (2025). Supervised vs. Unsupervised Pattern Recognition for SIEM
  False Positive Reduction. Pattern Recognition (under review).

GitHub: https://github.com/sumanshusohal/Supervised-vs.-Unsupervised-ML

Pipeline
--------
1. Data loading and preprocessing (CIC-IDS2017)
2. GridSearchCV hyperparameter optimisation with 3-fold CV
3. Training: Random Forest, XGBoost, Isolation Forest, Autoencoder
4. Evaluation: Accuracy, Precision, Recall, F1, FPR, FNR, AUC, Timing
5. Visualisation: ROC curves, Precision-Recall curves, Confusion matrices
6. Explainability: SHAP (tree models) + reconstruction error decomposition (AE)

Usage
-----
  python supervised_vs_unsupervised.py           # full run
  python supervised_vs_unsupervised.py --no-ae   # skip Autoencoder
"""

# ---------------------------------------------------------------------------- #
#  Auto-install missing dependencies                                            #
# ---------------------------------------------------------------------------- #
def _ensure(package, import_name=None):
    """Install *package* via pip if it cannot be imported."""
    import importlib, subprocess
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[setup] Installing {package} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package]
        )

import sys as _sys  # needed before setup runs

_ensure("tensorflow")
_ensure("shap")
_ensure("lime")
_ensure("imbalanced-learn", "imblearn")
_ensure("xgboost")
_ensure("scikit-learn", "sklearn")
_ensure("pandas")
_ensure("numpy")
_ensure("matplotlib")
_ensure("seaborn")
_ensure("joblib")
_ensure("requests")

# ---------------------------------------------------------------------------- #
#  Imports                                                                      #
# ---------------------------------------------------------------------------- #
import os, sys, time, warnings, traceback, argparse
import joblib, requests, zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
)

from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except ImportError:
    TF_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------- #
#  Paths & Constants                                                            #
# ---------------------------------------------------------------------------- #
BASE_DIR    = "experiment_results"
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR    = "CICIDS2017_data"
RESULTS_CSV = os.path.join(BASE_DIR, "model_comparison_results.csv")
PREPROCESSED = os.path.join(BASE_DIR, "preprocessed_data.joblib")

DATASET_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
)

# ---------------------------------------------------------------------------- #
#  Data Loading                                                                 #
# ---------------------------------------------------------------------------- #
def download_and_load_data():
    """Download CIC-IDS2017, extract, and return concatenated DataFrame."""
    zip_path = os.path.join(DATA_DIR, "MachineLearningCSV.zip")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {DATASET_URL} ...")
        try:
            r = requests.get(DATASET_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    # Extract
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

    # Load all CSVs
    frames = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.endswith(".csv"):
                path = os.path.join(root, fname)
                try:
                    df = pd.read_csv(path, encoding="latin1", low_memory=False)
                    frames.append(df)
                    print(f"  Loaded {fname}  shape={df.shape}")
                except Exception as e:
                    print(f"  Warning: could not load {fname}: {e}")

    if not frames:
        print("No CSV files loaded.")
        return None

    merged = pd.concat(frames, ignore_index=True)
    print(f"Merged dataset shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------- #
#  Preprocessing                                                                #
# ---------------------------------------------------------------------------- #
def preprocess_data(df):
    """
    Clean, encode, scale, split, and handle class imbalance.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series (scaled)
    feature_names                     : list[str]
    X_train_normal                    : benign-only training rows for AE
    """
    print("\n=== Preprocessing ===")
    df = df.copy()

    # Strip column whitespace
    df.columns = df.columns.str.strip()

    # Drop known non-informative columns
    drop_cols = ["Destination Port", "Fwd Header Length.1"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Handle infinities and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Rows removed (NaN/dup): {before - len(df):,}  |  Remaining: {len(df):,}")

    # Binary label encoding: 0 = Benign, 1 = Attack
    if "Label" not in df.columns:
        raise ValueError("Column 'Label' not found.")
    df["Label"] = df["Label"].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )
    print("Class distribution:\n", df["Label"].value_counts(normalize=True).to_string())

    X = df.drop(columns=["Label"])
    y = df["Label"]
    feature_names = list(X.columns)

    # 70 / 30 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Scale to [0, 1] — compatible with AE sigmoid output
    scaler = MinMaxScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    # Under-sample majority class in training set (supervised models only)
    # X_train_sc (full scaled) is kept for unsupervised models (IF, AE)
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_sc, y_train)
    print(f"After under-sampling — train: {X_train_res.shape}, test: {X_test_sc.shape}")

    # Benign-only subset for AE training
    X_train_normal = X_train_sc[y_train == 0]

    return X_train_res, X_test_sc, y_train_res, y_test, feature_names, X_train_normal, X_train_sc


# ---------------------------------------------------------------------------- #
#  Hyperparameter Grids                                                         #
# ---------------------------------------------------------------------------- #
RF_GRID = {
    "n_estimators":    [50, 100],
    "max_depth":       [10, 20],
    "min_samples_split": [2, 5],
}

XGB_GRID = {
    "n_estimators":  [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth":     [3, 5],
}


# ---------------------------------------------------------------------------- #
#  Supervised Training                                                          #
# ---------------------------------------------------------------------------- #
def train_random_forest(X_train, y_train):
    """GridSearchCV-optimised Random Forest (weighted F1 objective)."""
    print("\n--- Training Random Forest (GridSearchCV) ---")
    t0 = time.time()
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        RF_GRID, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Best params: {gs.best_params_}  |  Train time: {elapsed:.1f}s")
    return gs.best_estimator_, elapsed


def train_xgboost(X_train, y_train):
    """GridSearchCV-optimised XGBoost (weighted F1 objective)."""
    print("\n--- Training XGBoost (GridSearchCV) ---")
    t0 = time.time()
    gs = GridSearchCV(
        XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        XGB_GRID, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Best params: {gs.best_params_}  |  Train time: {elapsed:.1f}s")
    return gs.best_estimator_, elapsed


# ---------------------------------------------------------------------------- #
#  Unsupervised Training                                                        #
# ---------------------------------------------------------------------------- #
def train_isolation_forest(X_train_full):
    """
    Train Isolation Forest on the full (non-resampled) scaled training set.

    Parameters are fixed a priori from domain knowledge:
      contamination = 0.1  (conservative upper bound on attack fraction)
      n_estimators  = 100  (standard ensemble size)

    No labels are used at any point — this is a genuinely unsupervised procedure.
    X_train_full must be the complete scaled training set, NOT the under-sampled
    version used for supervised models.
    """
    print("\n--- Training Isolation Forest (unsupervised, fixed hyperparameters) ---")
    t0 = time.time()
    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_full)
    elapsed = time.time() - t0
    print(
        f"Isolation Forest trained on {X_train_full.shape[0]:,} samples "
        f"(contamination=0.1, n_estimators=100)  |  Train time: {elapsed:.1f}s"
    )
    return clf, elapsed


# ---------------------------------------------------------------------------- #
#  Autoencoder                                                                  #
# ---------------------------------------------------------------------------- #
def build_autoencoder(input_dim, latent_dim=8):
    """Symmetric feedforward autoencoder."""
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(16, activation="relu")(x)
    bottleneck = Dense(latent_dim, activation="relu")(x)
    x = Dense(16, activation="relu")(bottleneck)
    x = Dense(64, activation="relu")(x)
    out = Dense(input_dim, activation="sigmoid")(x)
    model = Model(inp, out)
    return model


def train_autoencoder(X_normal, input_dim, latent_dim=8, epochs=50, batch_size=512):
    """Train AE on benign-only data; return model, threshold, training time."""
    if not TF_OK:
        print("TensorFlow not available — skipping Autoencoder.")
        return None, None, 0

    print("\n--- Training Autoencoder ---")
    t0 = time.time()

    # Cap at 200k samples to avoid Colab timeout
    sample_size = min(len(X_normal), 200_000)
    X_samp = X_normal.sample(sample_size, random_state=42).values.astype(np.float32)

    ae = build_autoencoder(input_dim, latent_dim)
    ae.compile(optimizer="adam", loss="mse")
    ae.summary()

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ae.fit(
        X_samp, X_samp,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.2, callbacks=[es], verbose=1,
    )

    # Threshold = mean + 3 * std on normal training reconstruction errors
    recon = ae.predict(X_samp, batch_size=batch_size, verbose=0)
    errors = np.mean((X_samp - recon) ** 2, axis=1)
    threshold = float(np.mean(errors) + 3 * np.std(errors))

    elapsed = time.time() - t0
    print(f"AE threshold: {threshold:.6f}  |  Train time: {elapsed:.1f}s")
    return ae, threshold, elapsed


# ---------------------------------------------------------------------------- #
#  Evaluation                                                                   #
# ---------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score, model_name, train_time):
    """Return a dict of all paper metrics plus timing."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) else 0.0
    auc_score = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")

    return {
        "Model":       model_name,
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":    f1_score(y_true, y_pred, zero_division=0),
        "FPR":         fpr,
        "FNR":         fnr,
        "ROC-AUC":     auc_score,
        "Train Time (s)": train_time,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def evaluate_supervised(model, X_test, y_test, name, train_time):
    t0 = time.time()
    y_pred  = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    infer_time = (time.time() - t0) / (len(y_test) / 1000)
    metrics = compute_metrics(y_test, y_pred, y_score, name, train_time)
    metrics["Inference (s/1k)"] = round(infer_time, 4)
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    return metrics, y_pred, y_score


def evaluate_isolation_forest(model, X_test, y_test, train_time):
    t0 = time.time()
    raw    = model.predict(X_test)
    y_pred = np.where(raw == -1, 1, 0)
    y_score = -model.decision_function(X_test)
    infer_time = (time.time() - t0) / (len(y_test) / 1000)
    metrics = compute_metrics(y_test, y_pred, y_score, "Isolation Forest", train_time)
    metrics["Inference (s/1k)"] = round(infer_time, 4)
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    return metrics, y_pred, y_score


def evaluate_autoencoder(model, threshold, X_test, y_test, train_time):
    X_np = X_test.values.astype(np.float32)
    t0 = time.time()
    recon  = model.predict(X_np, batch_size=512, verbose=0)
    errors = np.mean((X_np - recon) ** 2, axis=1)
    y_pred = (errors > threshold).astype(int)
    infer_time = (time.time() - t0) / (len(y_test) / 1000)
    metrics = compute_metrics(y_test, y_pred, errors, "Autoencoder", train_time)
    metrics["Inference (s/1k)"] = round(infer_time, 4)
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    return metrics, y_pred, errors


# ---------------------------------------------------------------------------- #
#  Visualisation                                                                #
# ---------------------------------------------------------------------------- #
def _save(fig, fname):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_roc_pr(y_true, y_score, name):
    """ROC + Precision-Recall side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluation Curves — {name}", fontsize=13)

    fpr_v, tpr_v, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_v, tpr_v)
    ax1.plot(fpr_v, tpr_v, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC Curve", xlim=[0, 1], ylim=[0, 1.02])
    ax1.legend()
    ax1.grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    ax2.plot(rec, prec, lw=2, color="steelblue", label=f"AUC = {pr_auc:.4f}")
    ax2.set(xlabel="Recall", ylabel="Precision",
            title="Precision-Recall Curve", xlim=[0, 1], ylim=[0, 1.02])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    safe = name.replace(" ", "_")
    _save(fig, f"{safe}_roc_pr.png")


def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["BENIGN", "ATTACK"],
                yticklabels=["BENIGN", "ATTACK"], ax=ax)
    ax.set(title=f"Confusion Matrix — {name}",
           xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()
    safe = name.replace(" ", "_")
    _save(fig, f"{safe}_confusion.png")


def plot_summary_bar(results_df):
    """Bar chart comparing FPR and FNR across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, colour, title in [
        (axes[0], "FPR", "salmon",   "False Positive Rate (lower ↓ better)"),
        (axes[1], "FNR", "steelblue","False Negative Rate (lower ↓ better)"),
    ]:
        results_df[col].plot(kind="bar", ax=ax, color=colour, edgecolor="k")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.4)
        for patch in ax.patches:
            ax.annotate(
                f"{patch.get_height():.3f}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center", va="bottom", fontsize=9,
            )
    plt.tight_layout()
    _save(fig, "summary_fpr_fnr.png")


# ---------------------------------------------------------------------------- #
#  SHAP Explainability                                                          #
# ---------------------------------------------------------------------------- #
def explain_shap(model, X_sample_df, name):
    """SHAP summary plot (tree explainer) and save."""
    if not SHAP_OK:
        print("SHAP not installed — skipping.")
        return
    print(f"\n--- SHAP explanation: {name} ---")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_df)
        # For binary classification shap_values is a list; take class-1 array
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_sample_df, show=False)
        safe = name.replace(" ", "_")
        _save(plt.gcf(), f"{safe}_shap_summary.png")
    except Exception as e:
        print(f"SHAP error: {e}")


# ---------------------------------------------------------------------------- #
#  Main                                                                         #
# ---------------------------------------------------------------------------- #
def main(run_ae=True):
    # ── Colab: mount Drive ──────────────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        project_path = "/content/drive/My Drive/Colab_Projects/CIC_IDS_Analysis"
        os.makedirs(project_path, exist_ok=True)
        os.chdir(project_path)
        print(f"Working directory: {os.getcwd()}")
    except ImportError:
        pass

    for d in (BASE_DIR, MODELS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)

    # ── 1. Load & preprocess ────────────────────────────────────────────────
    if os.path.exists(PREPROCESSED):
        print("Loading preprocessed data from cache ...")
        X_train, X_test, y_train, y_test, features, X_normal, X_train_full = joblib.load(PREPROCESSED)
    else:
        raw = download_and_load_data()
        if raw is None:
            sys.exit("Data loading failed.")
        X_train, X_test, y_train, y_test, features, X_normal, X_train_full = preprocess_data(raw)
        joblib.dump(
            (X_train, X_test, y_train, y_test, features, X_normal, X_train_full),
            PREPROCESSED,
        )
        print("Preprocessed data cached.")

    input_dim   = X_train.shape[1]
    results     = []
    all_preds   = {}

    # ── 2. Random Forest ────────────────────────────────────────────────────
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if os.path.exists(rf_path):
        print("Loading cached Random Forest ...")
        rf, rf_time = joblib.load(rf_path)
    else:
        rf, rf_time = train_random_forest(X_train, y_train)
        joblib.dump((rf, rf_time), rf_path)

    m_rf, pred_rf, score_rf = evaluate_supervised(rf, X_test, y_test, "Random Forest", rf_time)
    results.append(m_rf)
    all_preds["Random Forest"] = (pred_rf, score_rf)
    plot_roc_pr(y_test, score_rf, "Random Forest")
    plot_confusion(y_test, pred_rf, "Random Forest")

    # ── 3. XGBoost ──────────────────────────────────────────────────────────
    xgb_path = os.path.join(MODELS_DIR, "xgboost.joblib")
    if os.path.exists(xgb_path):
        print("Loading cached XGBoost ...")
        xgb, xgb_time = joblib.load(xgb_path)
    else:
        xgb, xgb_time = train_xgboost(X_train, y_train)
        joblib.dump((xgb, xgb_time), xgb_path)

    m_xgb, pred_xgb, score_xgb = evaluate_supervised(xgb, X_test, y_test, "XGBoost", xgb_time)
    results.append(m_xgb)
    all_preds["XGBoost"] = (pred_xgb, score_xgb)
    plot_roc_pr(y_test, score_xgb, "XGBoost")
    plot_confusion(y_test, pred_xgb, "XGBoost")

    # ── 4. Isolation Forest ─────────────────────────────────────────────────
    if_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    if os.path.exists(if_path):
        print("Loading cached Isolation Forest ...")
        ifo, if_time = joblib.load(if_path)
    else:
        # Pass full scaled training set — IF is purely unsupervised, no labels used
        ifo, if_time = train_isolation_forest(X_train_full)
        joblib.dump((ifo, if_time), if_path)

    m_if, pred_if, score_if = evaluate_isolation_forest(ifo, X_test, y_test, if_time)
    results.append(m_if)
    all_preds["Isolation Forest"] = (pred_if, score_if)
    plot_roc_pr(y_test, score_if, "Isolation Forest")
    plot_confusion(y_test, pred_if, "Isolation Forest")

    # ── 5. Autoencoder ──────────────────────────────────────────────────────
    if run_ae and TF_OK:
        ae_path = os.path.join(MODELS_DIR, "autoencoder")
        if os.path.exists(ae_path + ".threshold"):
            print("Loading cached Autoencoder ...")
            from tensorflow.keras.models import load_model
            ae = load_model(ae_path)
            ae_thresh = float(open(ae_path + ".threshold").read())
            ae_time = 0.0
        else:
            ae, ae_thresh, ae_time = train_autoencoder(
                X_normal, input_dim, latent_dim=8, epochs=50, batch_size=512,
            )
            if ae is not None:
                ae.save(ae_path)
                with open(ae_path + ".threshold", "w") as f:
                    f.write(str(ae_thresh))

        if ae is not None:
            m_ae, pred_ae, score_ae = evaluate_autoencoder(ae, ae_thresh, X_test, y_test, ae_time)
            results.append(m_ae)
            all_preds["Autoencoder"] = (pred_ae, score_ae)
            plot_roc_pr(y_test, score_ae, "Autoencoder")
            plot_confusion(y_test, pred_ae, "Autoencoder")
    elif run_ae and not TF_OK:
        print("TensorFlow not available — Autoencoder skipped.")

    # ── 6. Results table ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    display_cols = ["Accuracy", "Precision", "Recall", "F1-Score",
                    "FPR", "FNR", "ROC-AUC", "Train Time (s)", "Inference (s/1k)"]
    print(results_df[[c for c in display_cols if c in results_df.columns]].to_string())
    results_df.to_csv(RESULTS_CSV)
    print(f"\nResults saved → {RESULTS_CSV}")

    plot_summary_bar(results_df)

    # ── 7. SHAP explanations ────────────────────────────────────────────────
    X_shap = X_test.sample(min(500, len(X_test)), random_state=42)
    explain_shap(rf,  X_shap, "Random Forest")
    explain_shap(xgb, X_shap, "XGBoost")

    print("\nPipeline complete.")


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised vs. Unsupervised ML for SIEM FP Reduction"
    )
    parser.add_argument("--no-ae", action="store_true",
                        help="Skip Autoencoder training")
    args = parser.parse_args()
    main(run_ae=not args.no_ae)

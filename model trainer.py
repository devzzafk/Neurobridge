"""
model_trainer.py
================
Reads the EMG dataset CSV, constructs the sliding-window feature matrix,
trains a Random Forest Classifier, prints a full terminal validation report,
and pickles the trained model to `emg_model.pkl`.

Run standalone:
    python model_trainer.py              # uses default emg_dataset.csv
    python model_trainer.py --generate   # generates fresh dataset first
"""

import argparse
import os
import pickle
import time
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Internal modules
from signal_processing import build_feature_matrix, FEATURE_NAMES, EMGFilterPipeline

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_FILE  = "emg_dataset.csv"
MODEL_FILE    = "emg_model.pkl"
CLASS_NAMES   = ["REST", "CLENCH", "FLICK"]
FS            = 1000        # Hz
WINDOW_MS     = 200.0       # ms
STEP_MS       = 50.0        # ms
RF_N_TREES    = 150
RF_MAX_DEPTH  = 12
RANDOM_STATE  = 42
TEST_SIZE     = 0.20        # 20% hold-out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_header(title: str):
    print()
    print(_divider("═"))
    print(f"  {title}")
    print(_divider("═"))


def _print_section(title: str):
    print()
    print(_divider())
    print(f"  {title}")
    print(_divider())


# ─── Data Loading & Feature Engineering ──────────────────────────────────────

def load_and_featurise(csv_path: str,
                       fs: int = FS,
                       window_ms: float = WINDOW_MS,
                       step_ms: float = STEP_MS,
                       apply_filter: bool = True,
                       verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CSV → optionally filter → sliding-window feature extraction.

    Returns
    -------
    X : feature matrix (n_windows, n_features)
    y : integer label vector (n_windows,)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: '{csv_path}'\n"
            "Run with --generate flag or call data_simulator.generate_dataset() first."
        )

    if verbose:
        print(f"[Trainer] Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"[Trainer] Rows: {len(df):,}  |  Columns: {list(df.columns)}")
        class_dist = df["label"].value_counts().sort_index()
        for cls, count in class_dist.items():
            print(f"          Class {cls} ({CLASS_NAMES[int(cls)]:7s}): "
                  f"{count:,} samples ({count/len(df)*100:.1f}%)")

    signal = df[["ch1", "ch2"]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    if apply_filter:
        if verbose:
            print("[Trainer] Applying EMG filter pipeline (BP + Notch)…")
        pipeline = EMGFilterPipeline(fs=fs)
        signal   = pipeline.process(signal)

    if verbose:
        print(f"[Trainer] Building feature matrix "
              f"(window={window_ms}ms, step={step_ms}ms)…")

    t0 = time.time()
    X, y = build_feature_matrix(signal, labels, fs=fs,
                                 window_ms=window_ms, step_ms=step_ms)
    elapsed = time.time() - t0

    if verbose:
        print(f"[Trainer] Feature matrix: {X.shape[0]:,} windows × "
              f"{X.shape[1]} features  ({elapsed:.2f}s)")

    return X, y


# ─── Model Training ───────────────────────────────────────────────────────────

def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        n_estimators: int = RF_N_TREES,
                        max_depth: int = RF_MAX_DEPTH,
                        verbose: bool = True) -> Pipeline:
    """
    Train a StandardScaler → RandomForestClassifier pipeline.

    Returns the fitted sklearn Pipeline.
    """
    if verbose:
        print(f"[Trainer] Training Random Forest "
              f"(n_estimators={n_estimators}, max_depth={max_depth})…")
    t0 = time.time()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ))
    ])
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(f"[Trainer] Training complete in {elapsed:.2f}s")

    return model


# ─── Validation Report ────────────────────────────────────────────────────────

def print_validation_report(model: Pipeline,
                             X_train: np.ndarray,
                             X_test: np.ndarray,
                             y_train: np.ndarray,
                             y_test: np.ndarray):
    """Print a comprehensive terminal validation report."""

    _print_header("SILENT SPEECH — MODEL VALIDATION REPORT")

    # ── Hold-out accuracy ──────────────────────────────────────────────────
    _print_section("Hold-out Test Set Performance")
    y_pred       = model.predict(X_test)
    test_acc     = accuracy_score(y_test, y_pred)
    train_acc    = accuracy_score(y_train, model.predict(X_train))

    print(f"  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  Accuracy : {test_acc  * 100:.2f}%")
    print(f"  Overfitting Δ  : {(train_acc - test_acc) * 100:+.2f}%")

    # ── Classification report ─────────────────────────────────────────────
    _print_section("Per-Class Precision / Recall / F1")
    report = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES,
                                   digits=4)
    # Indent each line
    for line in report.splitlines():
        print(f"  {line}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    _print_section("Confusion Matrix  (rows=actual, cols=predicted)")
    cm = confusion_matrix(y_test, y_pred)
    header = "          " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8,}" for v in row)
        print(f"  {CLASS_NAMES[i]:8s}  {row_str}")

    # ── Cross-validation ──────────────────────────────────────────────────
    _print_section("5-Fold Stratified Cross-Validation")
    print("  Running… (this may take a moment)")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  Fold accuracies: {[f'{s*100:.2f}%' for s in scores]}")
    print(f"  Mean  ± StdDev : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    # ── Feature importances ───────────────────────────────────────────────
    _print_section("Feature Importances (Random Forest)")
    rf          = model.named_steps["rf"]
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        bar  = "█" * int(importances[idx] * 80)
        print(f"  {rank+1:2d}. {name:12s}  {importances[idx]:.4f}  {bar}")

    # ── Model metadata ────────────────────────────────────────────────────
    _print_section("Model Metadata")
    print(f"  Algorithm     : Random Forest Classifier")
    print(f"  N Estimators  : {rf.n_estimators}")
    print(f"  Max Depth     : {rf.max_depth}")
    print(f"  N Features    : {X_train.shape[1]}")
    print(f"  Feature Names : {FEATURE_NAMES}")
    print(f"  Classes       : {CLASS_NAMES}")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")
    print()
    print(_divider("═"))
    print()


# ─── Model Persistence ────────────────────────────────────────────────────────

def save_model(model: Pipeline,
               metadata: dict,
               path: str = MODEL_FILE,
               verbose: bool = True):
    """Pickle model + metadata dict to disk."""
    payload = {
        "model"        : model,
        "metadata"     : metadata,
        "class_names"  : CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "fs"           : FS,
        "window_ms"    : WINDOW_MS,
        "step_ms"      : STEP_MS,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        size_kb = os.path.getsize(path) / 1024
        print(f"[Trainer] Model saved → '{path}'  ({size_kb:.1f} KB)")


def load_model(path: str = MODEL_FILE) -> tuple[Pipeline, dict]:
    """
    Load pickled model payload.

    Returns
    -------
    model    : fitted sklearn Pipeline
    metadata : dict with training metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: '{path}'")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload


# ─── Main Training Routine ────────────────────────────────────────────────────

def train(csv_path: str = DATASET_FILE,
          model_path: str = MODEL_FILE,
          verbose: bool = True) -> Pipeline:
    """
    End-to-end training pipeline:
      load → featurise → split → train → validate → save

    Returns the fitted model Pipeline.
    """
    # 1. Load & featurise
    X, y = load_and_featurise(csv_path, verbose=verbose)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 3. Train
    model = train_random_forest(X_train, y_train, verbose=verbose)

    # 4. Validation report
    if verbose:
        print_validation_report(model, X_train, X_test, y_train, y_test)

    # 5. Save
    y_pred   = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    metadata = {
        "test_accuracy"  : test_acc,
        "train_accuracy" : accuracy_score(y_train, model.predict(X_train)),
        "n_train"        : int(len(X_train)),
        "n_test"         : int(len(X_test)),
        "trained_at"     : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_model(model, metadata, model_path, verbose=verbose)

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Silent Speech TinyML EMG gesture classifier."
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate a fresh dataset before training."
    )
    parser.add_argument(
        "--dataset", default=DATASET_FILE,
        help=f"Path to training CSV (default: {DATASET_FILE})"
    )
    parser.add_argument(
        "--model", default=MODEL_FILE,
        help=f"Output model path (default: {MODEL_FILE})"
    )
    args = parser.parse_args()

    if args.generate:
        print("[Trainer] --generate flag detected. Generating dataset first…")
        from data_simulator import generate_dataset
        generate_dataset(verbose=True)

    train(csv_path=args.dataset, model_path=args.model, verbose=True)

"""
Veritas — ML Anomaly Detection Training Pipeline
══════════════════════════════════════════════════
Trains an Isolation Forest (unsupervised) on synthetic institutional
transaction data and persists the full inference pipeline for the
Flask /predict endpoint.

Saved Artifacts (ml/models/):
  ├── isolation_forest.joblib   — Trained IsolationForest model
  ├── scaler.joblib             — Fitted StandardScaler (numerical features)
  ├── encoder_columns.joblib    — Column names after one-hot encoding
  └── feature_config.joblib     — Feature engineering metadata

Usage:
    python -m ml.train_model               # from project root
    python ml/train_model.py               # direct execution
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ─── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "transactions.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Columns to drop before training (identifiers + labels)
DROP_COLS = [
    "transaction_id",
    "from_account",
    "to_account",
    "is_synthetic_anomaly",
    "anomaly_type",
]

# Feature groups
NUMERICAL_FEATURES = ["amount"]
CATEGORICAL_FEATURES = ["transaction_type", "currency", "region"]
TEMPORAL_FEATURE = "timestamp"

# IsolationForest hyperparameters
IF_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.07,      # ~7.4% anomaly rate in our synthetic data
    "max_samples": "auto",
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 0,
}

SEED = 42


# ─── Utilities ─────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _step(msg: str) -> None:
    """Print a pipeline step."""
    print(f"  ▸ {msg}")


# ─── Step 1: Data Loading ─────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the raw transaction CSV."""

    _header("STEP 1 — Data Loading")

    if not os.path.exists(path):
        print(f"  ✘ File not found: {path}")
        print("    Run `python -m data.synthetic_data_gen` first.")
        sys.exit(1)

    df = pd.read_csv(path).copy()
    _step(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {os.path.basename(path)}")
    _step(f"Anomaly rate: {df['is_synthetic_anomaly'].mean():.2%} "
          f"({df['is_synthetic_anomaly'].sum()} anomalous)")

    return df


# ─── Step 2: Feature Engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract temporal features from timestamp and prepare the feature matrix.
    Returns (feature_df, list_of_final_column_names).
    """

    _header("STEP 2 — Feature Engineering")

    # Parse timestamp into a datetime column on a fresh copy
    ts = pd.to_datetime(df["timestamp"])

    # ── Temporal + derived features (assign() avoids chained-assignment) ──
    df = (
        df.assign(
            timestamp=ts,
            hour=ts.dt.hour,
            day_of_week=ts.dt.dayofweek,              # Mon=0 … Sun=6
            is_weekend=(ts.dt.dayofweek >= 5).astype(int),
            is_night=((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int),
            amount_log=np.log1p(df["amount"]),
        )
    )

    temporal_features = ["hour", "day_of_week", "is_weekend", "is_night"]
    _step(f"Extracted temporal features: {temporal_features}")
    _step("Applied log1p transform → amount_log")

    # ── One-hot encode categoricals ────────────────────────────────────
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    _step(f"One-hot encoded: {CATEGORICAL_FEATURES}")

    # ── Select final feature columns ───────────────────────────────────
    # Keep: amount, amount_log, temporal features, and all OHE columns
    ohe_cols = [c for c in df_encoded.columns
                if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]

    feature_cols = (
        NUMERICAL_FEATURES +
        ["amount_log"] +
        temporal_features +
        sorted(ohe_cols)
    )

    _step(f"Total features: {len(feature_cols)}")
    _step(f"Feature vector sample: {feature_cols[:6]} …")

    return df_encoded, feature_cols


# ─── Step 3: Preprocessing (Scaling) ──────────────────────────────────────────

def scale_features(
    X: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """Scale numerical features using StandardScaler."""

    _header("STEP 3 — Feature Scaling")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    _step(f"Fitted StandardScaler on {X_scaled.shape[1]} features")
    _step(f"Training matrix shape: {X_scaled.shape}")

    return X_scaled, scaler


# ─── Step 4: Model Training ───────────────────────────────────────────────────

def train_isolation_forest(X_scaled: np.ndarray) -> IsolationForest:
    """Train an Isolation Forest on the scaled feature matrix."""

    _header("STEP 4 — Isolation Forest Training")

    _step(f"Hyperparameters: {IF_PARAMS}")

    model = IsolationForest(**IF_PARAMS)

    start_time = time.time()
    model.fit(X_scaled)
    elapsed = time.time() - start_time

    _step(f"Training complete in {elapsed:.2f}s")
    _step(f"Trees: {model.n_estimators}  |  Contamination: {model.contamination}")

    return model


# ─── Step 5: Evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    model: IsolationForest,
    X_scaled: np.ndarray,
    y_true: pd.Series,
) -> np.ndarray:
    """
    Predict on the training set and evaluate against ground-truth labels.
    IsolationForest returns: +1 = normal, -1 = anomaly.
    """

    _header("STEP 5 — Model Evaluation")

    # Raw predictions
    raw_preds = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    # Convert IF convention → boolean (True = anomaly)
    y_pred = (raw_preds == -1)
    y_actual = y_true.astype(bool).values

    # ── Classification Report ──────────────────────────────────────────
    target_names = ["Normal", "Anomaly"]
    report = classification_report(
        y_actual, y_pred, target_names=target_names, digits=4
    )
    print(f"\n  Classification Report:\n")
    for line in report.split("\n"):
        print(f"    {line}")

    # ── Confusion Matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Normal    Predicted Anomaly")
    print(f"    Actual Normal   {tn:>10,}          {fp:>10,}")
    print(f"    Actual Anomaly  {fn:>10,}          {tp:>10,}")

    # ── Key Metrics Summary ────────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_actual, y_pred, average="binary", pos_label=True
    )

    print(f"\n  Key Anomaly Detection Metrics:")
    print(f"    Precision (anomaly) : {precision:.4f}")
    print(f"    Recall (anomaly)    : {recall:.4f}")
    print(f"    F1-Score (anomaly)  : {f1:.4f}")
    print(f"    True Positives      : {tp:,}")
    print(f"    False Positives     : {fp:,}")
    print(f"    Missed Anomalies    : {fn:,}")

    # ── Anomaly Score Distribution ─────────────────────────────────────
    print(f"\n  Anomaly Score Distribution:")
    print(f"    Normal  mean={anomaly_scores[~y_actual].mean():.4f}  "
          f"std={anomaly_scores[~y_actual].std():.4f}")
    print(f"    Anomaly mean={anomaly_scores[y_actual].mean():.4f}  "
          f"std={anomaly_scores[y_actual].std():.4f}")

    return y_pred


# ─── Step 6: Model Persistence ────────────────────────────────────────────────

def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> None:
    """Persist the trained model and preprocessing artifacts."""

    _header("STEP 6 — Saving Artifacts")

    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {
        "isolation_forest.joblib": model,
        "scaler.joblib": scaler,
        "encoder_columns.joblib": feature_cols,
        "feature_config.joblib": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "temporal_feature": TEMPORAL_FEATURE,
            "drop_cols": DROP_COLS,
            "if_params": IF_PARAMS,
        },
    }

    for filename, artifact in artifacts.items():
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(artifact, path)
        size_kb = os.path.getsize(path) / 1024
        _step(f"Saved {filename:<28s} ({size_kb:>8.1f} KB)")

    print(f"\n  📁 All artifacts saved to: {MODEL_DIR}/")


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full ML training pipeline."""

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Veritas — Anomaly Detection Training Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Feature engineering
    df_engineered, feature_cols = engineer_features(df)

    # 3. Prepare X and y
    X = df_engineered[feature_cols]
    y_true = df_engineered["is_synthetic_anomaly"]

    # 4. Scale features
    X_scaled, scaler = scale_features(X, feature_cols)

    # 5. Train Isolation Forest
    model = train_isolation_forest(X_scaled)

    # 6. Evaluate against ground truth
    evaluate_model(model, X_scaled, y_true)

    # 7. Persist artifacts
    save_artifacts(model, scaler, feature_cols)

    print("\n  🚀 Phase 2 complete. Model ready for Flask /predict endpoint.\n")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

"""
Veritas — ML Anomaly Detection Training Pipeline
═════════════════════════════════════════════════
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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ─── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "ml_feature_matrix.parquet"
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

# Regions known to carry elevated AML/sanctions risk
HIGH_RISK_REGIONS = [
    "Bermuda", "British Virgin Islands", "Cayman Islands", "Crimea",
    "Iran", "Isle of Man", "North Korea", "Panama", "Syria",
]

# IsolationForest hyperparameters
IF_PARAMS = {
    "n_estimators": 300,
    "contamination": 0.074,     # matched to actual 7.41% anomaly rate
    "max_samples": 256,         # smaller sub-samples → sharper isolation splits
    "max_features": 0.8,        # feature sub-sampling adds tree diversity
    "random_state": 42,
    "bootstrap": True,          # bootstrap sampling for better generalization
    "n_jobs": -1,
    "verbose": 0,
}

# RandomForest supervised classifier hyperparameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "class_weight": "balanced",  # up-weight the 7.4% minority class
    "random_state": 42,
    "n_jobs": -1,
}

# Ensemble weight: how much to trust the supervised RF vs unsupervised IF
# Higher → more weight on RF (which learns labeled patterns directly)
ENSEMBLE_RF_WEIGHT = 0.7

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
    """Load the pre-engineered Parquet feature matrix."""

    _header("STEP 1 — Data Loading")

    if not os.path.exists(path):
        print(f"  ✘ File not found: {path}")
        print("    Ensure the Databricks pipeline has exported ml_feature_matrix.parquet.")
        sys.exit(1)

    df = pd.read_parquet(path).copy()
    _step(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {os.path.basename(path)}")
    _step(f"Anomaly rate: {df['is_synthetic_anomaly'].mean():.2%} "
          f"({df['is_synthetic_anomaly'].sum()} anomalous)")

    return df


# ─── Step 2: Feature Engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare the feature matrix from the pre-engineered Parquet dataset.
    Temporal and derived features (hour, day_of_week, is_weekend, is_night,
    amount_log) are expected to already exist in the DataFrame.
    Adds statistical and domain-driven features for stronger anomaly signal.
    Returns (feature_df, list_of_final_column_names).
    """

    _header("STEP 2 — Feature Engineering")

    # ── Derive is_night from hour if the Databricks pipeline didn't include it
    if "is_night" not in df.columns:
        df = df.assign(is_night=((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int))
        _step("Derived is_night from hour column (not in Parquet)")

    temporal_features = ["hour", "day_of_week", "is_weekend", "is_night"]
    _step(f"Pre-computed temporal features present: {temporal_features}")
    _step("Pre-computed amount_log present")

    # ── Statistical amount features ────────────────────────────────────
    # Z-score: how many standard deviations from the mean
    amt_mean = df["amount"].mean()
    amt_std = df["amount"].std()
    df = df.assign(
        amount_zscore=(df["amount"] - amt_mean) / amt_std,
        amount_percentile=df["amount"].rank(pct=True),
    )
    _step("Computed amount_zscore and amount_percentile")

    # ── Structuring detection ──────────────────────────────────────────
    # Structuring = splitting amounts to stay just below regulatory
    # reporting thresholds ($10,000 in the US). Distance to threshold
    # is a strong indicator: structuring txns cluster at ~$9,995.
    REPORTING_THRESHOLD = 10_000
    df = df.assign(
        amt_dist_to_threshold=np.abs(df["amount"] - REPORTING_THRESHOLD),
        is_near_threshold=((df["amount"] >= 9_000) & (df["amount"] < REPORTING_THRESHOLD)).astype(int),
    )
    _step(f"Structuring features: {df['is_near_threshold'].sum()} txns near $10K threshold")

    # ── High-value flag ────────────────────────────────────────────────
    # Offshore sweeps and geo-mismatch txns are $270K–$1.7M
    df = df.assign(
        is_high_value=(df["amount"] >= 200_000).astype(int),
    )
    _step(f"High-value flag (≥$200K): {df['is_high_value'].sum()} txns")

    # ── Account velocity ──────────────────────────────────────────────
    # Number of transactions per from_account (velocity spikes = many
    # micro-txns from the same account)
    acct_freq = df.groupby("from_account").size().rename("acct_txn_count")
    df = df.merge(acct_freq, on="from_account", how="left")
    _step(f"Account velocity: mean={df['acct_txn_count'].mean():.1f}, max={df['acct_txn_count'].max()}")

    # ── Domain-driven risk flag ────────────────────────────────────────
    # Consolidate 9 offshore/sanctioned region OHE columns into one signal
    df = df.assign(
        is_high_risk_region=df["region"].isin(HIGH_RISK_REGIONS).astype(int)
    )
    _step(f"Flagged {df['is_high_risk_region'].sum()} txns in {len(HIGH_RISK_REGIONS)} high-risk regions")

    # ── Interaction features ───────────────────────────────────────────
    df = df.assign(
        risk_night_interaction=df["is_high_risk_region"] * df["is_night"],
        risk_highval_interaction=df["is_high_risk_region"] * df["is_high_value"],
    )
    _step("Created risk × night and risk × high-value interaction features")

    # ── One-hot encode categoricals ────────────────────────────────────
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    _step(f"One-hot encoded: {CATEGORICAL_FEATURES}")

    # ── Select final feature columns ───────────────────────────────────
    ohe_cols = [c for c in df_encoded.columns
                if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]

    derived_features = [
        "amount_zscore", "amount_percentile",
        "amt_dist_to_threshold", "is_near_threshold",
        "is_high_value", "acct_txn_count",
        "is_high_risk_region", "risk_night_interaction", "risk_highval_interaction",
    ]

    feature_cols = (
        NUMERICAL_FEATURES +
        ["amount_log"] +
        derived_features +
        temporal_features +
        sorted(ohe_cols)
    )

    _step(f"Total features: {len(feature_cols)}")
    _step(f"Feature vector: {feature_cols}")

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

    _header("STEP 4a — Isolation Forest Training")

    _step(f"Hyperparameters: {IF_PARAMS}")

    model = IsolationForest(**IF_PARAMS)

    start_time = time.time()
    model.fit(X_scaled)
    elapsed = time.time() - start_time

    _step(f"Training complete in {elapsed:.2f}s")
    _step(f"Trees: {model.n_estimators}  |  Contamination: {model.contamination}")

    return model


def train_random_forest(
    X_scaled: np.ndarray,
    y_true: np.ndarray,
) -> RandomForestClassifier:
    """Train a supervised Random Forest for anomaly classification."""

    _header("STEP 4b — Supervised Random Forest Training")

    _step(f"Hyperparameters: {RF_PARAMS}")
    _step(f"Class distribution — Normal: {(y_true == 0).sum():,}  Anomaly: {(y_true == 1).sum():,}")

    rf = RandomForestClassifier(**RF_PARAMS)

    start_time = time.time()
    rf.fit(X_scaled, y_true)
    elapsed = time.time() - start_time

    _step(f"Training complete in {elapsed:.2f}s")
    _step(f"OOB-style train accuracy: {rf.score(X_scaled, y_true):.4f}")

    # Feature importance (top 10)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    _step("Top-10 feature importances:")
    # We'll print these when feature names are available — defer to eval

    return rf


# ─── Step 5: Evaluation ───────────────────────────────────────────────────────

def _print_eval_block(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> tuple[float, float, float]:
    """Print classification metrics for a given prediction set. Returns (precision, recall, f1)."""

    target_names = ["Normal", "Anomaly"]
    report = classification_report(
        y_actual, y_pred, target_names=target_names, digits=4
    )
    print(f"\n  {label} — Classification Report:\n")
    for line in report.split("\n"):
        print(f"    {line}")

    cm = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Normal    Predicted Anomaly")
    print(f"    Actual Normal   {tn:>10,}          {fp:>10,}")
    print(f"    Actual Anomaly  {fn:>10,}          {tp:>10,}")

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

    return precision, recall, f1


def evaluate_model(
    if_model: IsolationForest,
    rf_model: RandomForestClassifier,
    X_scaled: np.ndarray,
    y_true: pd.Series,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    """
    Evaluate both models and an ensemble, find the optimal threshold.
    Returns (y_pred_final, best_threshold).
    """

    _header("STEP 5 — Model Evaluation")

    y_actual = y_true.astype(bool).values

    # ── A) Isolation Forest only ──────────────────────────────────────
    if_scores = if_model.decision_function(X_scaled)  # lower = more anomalous
    raw_preds = if_model.predict(X_scaled)
    y_pred_if = (raw_preds == -1)
    _print_eval_block(y_actual, y_pred_if, "ISOLATION FOREST (default)")

    # ── B) Random Forest only ─────────────────────────────────────────
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]  # P(anomaly)
    y_pred_rf = (rf_probs >= 0.5)
    _print_eval_block(y_actual, y_pred_rf, "RANDOM FOREST (P≥0.5)")

    # Print top feature importances from RF
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  RF Top-10 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:>2}. {feature_cols[idx]:<30s}  {importances[idx]:.4f}")

    # ── C) Ensemble: weighted combination ─────────────────────────────
    # Normalize IF scores to [0, 1] range (invert so higher = more anomalous)
    if_anomaly_score = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
    ensemble_score = (ENSEMBLE_RF_WEIGHT * rf_probs +
                      (1 - ENSEMBLE_RF_WEIGHT) * if_anomaly_score)

    _step(f"Ensemble weights: RF={ENSEMBLE_RF_WEIGHT}, IF={1-ENSEMBLE_RF_WEIGHT}")

    # Threshold search on ensemble scores
    _step("Searching for optimal ensemble threshold …")

    best_f1 = 0.0
    best_threshold = 0.5
    thresholds = np.arange(0.05, 0.95, 0.005)

    for thr in thresholds:
        y_thr = ensemble_score >= thr
        if y_thr.sum() == 0:
            continue
        _, _, f1_thr, _ = precision_recall_fscore_support(
            y_actual, y_thr, average="binary", pos_label=True, zero_division=0
        )
        if f1_thr > best_f1:
            best_f1 = f1_thr
            best_threshold = thr

    _step(f"Best ensemble threshold: {best_threshold:.4f}  →  F1 = {best_f1:.4f}")

    y_pred_ensemble = ensemble_score >= best_threshold
    _print_eval_block(y_actual, y_pred_ensemble, "ENSEMBLE (optimized)")

    # ── Anomaly Score Distribution ────────────────────────────────────
    print(f"\n  Ensemble Score Distribution:")
    print(f"    Normal  mean={ensemble_score[~y_actual].mean():.4f}  "
          f"std={ensemble_score[~y_actual].std():.4f}")
    print(f"    Anomaly mean={ensemble_score[y_actual].mean():.4f}  "
          f"std={ensemble_score[y_actual].std():.4f}")
    print(f"    Separation: {ensemble_score[y_actual].mean() - ensemble_score[~y_actual].mean():.4f}")

    return y_pred_ensemble, best_threshold


# ─── Step 6: Model Persistence ────────────────────────────────────────────────

def save_artifacts(
    if_model: IsolationForest,
    rf_model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_cols: list[str],
    best_threshold: float,
) -> None:
    """Persist the trained models and preprocessing artifacts."""

    _header("STEP 6 — Saving Artifacts")

    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {
        "isolation_forest.joblib": if_model,
        "random_forest.joblib": rf_model,
        "scaler.joblib": scaler,
        "encoder_columns.joblib": feature_cols,
        "threshold.joblib": best_threshold,
        "feature_config.joblib": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "temporal_feature": TEMPORAL_FEATURE,
            "high_risk_regions": HIGH_RISK_REGIONS,
            "drop_cols": DROP_COLS,
            "if_params": IF_PARAMS,
            "rf_params": RF_PARAMS,
            "ensemble_rf_weight": ENSEMBLE_RF_WEIGHT,
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

    # 5a. Train Isolation Forest (unsupervised)
    if_model = train_isolation_forest(X_scaled)

    # 5b. Train Random Forest (supervised)
    rf_model = train_random_forest(X_scaled, y_true.values)

    # 6. Evaluate both models + ensemble + find optimal threshold
    _, best_threshold = evaluate_model(
        if_model, rf_model, X_scaled, y_true, feature_cols
    )

    # 7. Persist all artifacts
    save_artifacts(if_model, rf_model, scaler, feature_cols, best_threshold)

    print("\n  🚀 Phase 2 complete. Ensemble model ready for Flask /predict endpoint.\n")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
"""

 Veritas Financial Risk & Advisory Platform
 F lask Backend API

Endpoints:
  GET  /                         Health check
  POST /api/v1/ingest            Per sist raw transactions
  POST /api /v1 /predict           Score   with IsolationForest
  POST /api/v1/search            Retrieve regulatory context
  POST /api/v1/advise            Run  the Virtual Risk Committee
"""

# ── SQLite fix for Azure App Service ───────────────────────────────────────────
# ChromaDB (pulled by CrewAI) requires SQLite >= 3.35.0 but Azure's Python image
# ships with an older version. pysqlite3-binary is a drop-in replacement.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Not on Azure — system sqlite3 is fine

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import Blueprint, Flask, jsonify, render_template, request, session, redirect, url_for, flash
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import get_collections, init_db

load_dotenv()
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
LOCAL_REGULATORY_DOCS = PROJECT_ROOT / "data" / "processed" / "regulatory_embeddings.json"

INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "veritas-regulations")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")

CATEGORICAL_FEATURES = ["transaction_type", "currency", "region"]
TEMPORAL_FEATURES = ["hour", "day_of_week", "is_weekend", "is_night"]

# Regions matching the training pipeline's HIGH_RISK_REGIONS list
DEFAULT_HIGH_RISK_REGIONS = [
    "Bermuda", "British Virgin Islands", "Cayman Islands", "Crimea",
    "Iran", "Isle of Man", "North Korea", "Panama", "Syria",
]

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")
_model_cache: dict[str, Any] = {}
_embedding_model = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonify_value(value):
    """Convert MongoDB, datetime, and numpy values into JSON-safe objects."""

    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonify_value(item) for item in value]
    return value


def _request_json():
    payload = request.get_json(silent=True)
    if payload is None:
        return None, (jsonify({"error": "Request body must be valid JSON."}), 400)
    return payload, None


DEFAULT_SETTINGS = {
    "general": {
        "platform_name": "Veritas Financial Advisory",
        "timezone": "UTC (Coordinated Universal Time)",
        "default_currency": "USD — US Dollar",
        "alert_amount_threshold": 0,
    },
    "ml_engine": {
        "ensemble_rf_weight": 0.7,
        "anomaly_threshold": 0.52,
        "high_risk_regions": DEFAULT_HIGH_RISK_REGIONS,
    },
    "api_keys": {
        "groq_api_key": "gsk_xxxxxxxxxxxxxxxxxxxx",
        "azure_ai_search_endpoint": "https://veritas-search-*.search.windows.net",
        "mongodb_connection": "mongodb+srv://veritas:*****@cluster.mongodb.net",
    },
}


def _merge_settings(overrides):
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    if not isinstance(overrides, dict):
        return merged

    for section, values in overrides.items():
        if section not in merged or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(merged[section].get(key), dict) and isinstance(value, dict):
                merged[section][key].update(value)
            else:
                merged[section][key] = value
    return merged


def _normalize_records(payload):
    """Accept a single JSON object, a JSON list, or {"transactions": [...]}."""

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("transactions"), list):
        records = payload["transactions"]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError("Payload must be an object, list, or {'transactions': [...]} object.")

    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("Transaction payload must contain one or more JSON objects.")

    return records


def _load_model_artifacts():
    if _model_cache:
        return _model_cache

    required_files = {
        "if_model": MODEL_DIR / "isolation_forest.joblib",
        "rf_model": MODEL_DIR / "random_forest.joblib",
        "scaler": MODEL_DIR / "scaler.joblib",
        "feature_columns": MODEL_DIR / "encoder_columns.joblib",
        "feature_config": MODEL_DIR / "feature_config.joblib",
        "threshold": MODEL_DIR / "threshold.joblib",
    }

    # rf_model and threshold are optional (backward compat with IF-only artifacts)
    optional = {"rf_model", "threshold"}
    missing = [
        str(path) for name, path in required_files.items()
        if not path.exists() and name not in optional
    ]
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {missing}")

    for name, path in required_files.items():
        if path.exists():
            _model_cache[name] = joblib.load(path)

    # Backward compat: keep 'model' key pointing to IF for legacy callers
    _model_cache["model"] = _model_cache["if_model"]
    return _model_cache


def _get_user_settings():
    """Load the current user's settings from DB, merged with defaults."""
    user_id = session.get("user_id")
    if not user_id:
        return DEFAULT_SETTINGS.copy()
    try:
        users = get_collections()["users"]
        user = users.find_one({"_id": ObjectId(user_id)}, {"settings": 1})
        return _merge_settings((user or {}).get("settings"))
    except Exception:
        return DEFAULT_SETTINGS.copy()


def _prepare_feature_matrix(records, high_risk_regions=None):
    """Mirror the training feature engineering pipeline from train_model.py."""
    if high_risk_regions is None:
        high_risk_regions = DEFAULT_HIGH_RISK_REGIONS

    df = pd.DataFrame(records).copy()

    if "amount" not in df.columns:
        raise ValueError("Each transaction must include an 'amount' field.")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().any():
        raise ValueError("'amount' must be numeric for every transaction.")

    if "timestamp" not in df.columns:
        df["timestamp"] = _utc_now().isoformat()

    for column, default in {
        "transaction_type": "ACH",
        "currency": "USD",
        "region": "North America",
    }.items():
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default).astype(str)

    # Ensure from_account exists for velocity calculation
    if "from_account" not in df.columns:
        df["from_account"] = "unknown"

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    ts = ts.fillna(pd.Timestamp(_utc_now()))

    # ── Temporal features ─────────────────────────────────────────
    df = df.assign(
        hour=ts.dt.hour,
        day_of_week=ts.dt.dayofweek,
        is_weekend=(ts.dt.dayofweek >= 5).astype(int),
        is_night=((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int),
        amount_log=np.log1p(df["amount"]),
    )

    # ── Statistical features ──────────────────────────────────────
    amt_mean = df["amount"].mean()
    amt_std = df["amount"].std() if len(df) > 1 else 1.0
    df = df.assign(
        amount_zscore=(df["amount"] - amt_mean) / max(amt_std, 1e-9),
        amount_percentile=df["amount"].rank(pct=True),
    )

    # ── Structuring detection ─────────────────────────────────────
    REPORTING_THRESHOLD = 10_000
    df = df.assign(
        amt_dist_to_threshold=np.abs(df["amount"] - REPORTING_THRESHOLD),
        is_near_threshold=((df["amount"] >= 9_000) & (df["amount"] < REPORTING_THRESHOLD)).astype(int),
    )

    # ── High-value flag ───────────────────────────────────────────
    df = df.assign(is_high_value=(df["amount"] >= 200_000).astype(int))

    # ── Account velocity ──────────────────────────────────────────
    acct_freq = df.groupby("from_account").size().rename("acct_txn_count")
    df = df.merge(acct_freq, on="from_account", how="left")

    # ── High-risk region flag ─────────────────────────────────────
    df = df.assign(is_high_risk_region=df["region"].isin(high_risk_regions).astype(int))

    # ── Interaction features ──────────────────────────────────────
    df = df.assign(
        risk_night_interaction=df["is_high_risk_region"] * df["is_night"],
        risk_highval_interaction=df["is_high_risk_region"] * df["is_high_value"],
    )

    # ── One-hot encode ────────────────────────────────────────────
    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    feature_columns = _load_model_artifacts()["feature_columns"]
    features = encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = _load_model_artifacts()["scaler"].transform(features)

    return scaled


def _build_alert(record, anomaly_score, ensemble_score):
    """Build an alert document for an anomalous transaction."""
    alert_transaction = {
        **record,
        "transaction_id": record.get("transaction_id") or str(uuid.uuid4()),
        "timestamp": record.get("timestamp") or _utc_now().isoformat(),
        "currency": record.get("currency") or "USD",
        "transaction_type": record.get("transaction_type") or "ACH",
        "region": record.get("region") or "North America",
    }

    return {
        "transaction": alert_transaction,
        "transaction_id": alert_transaction["transaction_id"],
        "anomaly_score": float(anomaly_score),
        "ensemble_score": float(ensemble_score),
        "prediction": -1,
        "status": "open",
        "created_at": _utc_now(),
        "source": "ml_ensemble",
    }


def _get_dashboard_snapshot():
    collections = get_collections()

    total_transactions = collections["transactions"].count_documents({})
    total_alerts = collections["alerts"].count_documents({})
    open_alerts = collections["alerts"].count_documents({"status": "open"})

    if total_alerts > 0:
        risk_score = min(int((open_alerts / max(total_alerts, 1)) * 100), 100)
    else:
        risk_score = 0

    risk_level = (
        "High" if risk_score >= 65
        else "Medium" if risk_score >= 35
        else "Low"
    )

    recent_cursor = (
        collections["transactions"]
        .find()
        .sort("_id", -1)
        .limit(5)
    )
    recent_transactions = [_jsonify_value(doc) for doc in recent_cursor]

    return {
        "metrics": {
            "total_transactions": total_transactions,
            "total_alerts": total_alerts,
            "open_alerts": open_alerts,
            "risk_score": risk_score,
            "risk_level": risk_level,
        },
        "recent_transactions": recent_transactions,
        "chart_data": _build_risk_trend_data(collections),
    }


def _coerce_datetime(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    return None


def _bucket_start(dt: datetime, range_key: str) -> datetime:
    if range_key == "daily":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if range_key == "weekly":
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day - timedelta(days=start_of_day.weekday())
    if range_key == "monthly":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported range key: {range_key}")


def _next_bucket_start(dt: datetime, range_key: str) -> datetime:
    if range_key == "daily":
        return dt + timedelta(days=1)
    if range_key == "weekly":
        return dt + timedelta(weeks=1)
    if range_key == "monthly":
        if dt.month == 12:
            return dt.replace(year=dt.year + 1, month=1)
        return dt.replace(month=dt.month + 1)
    raise ValueError(f"Unsupported range key: {range_key}")


def _format_bucket_label(dt: datetime, range_key: str) -> str:
    if range_key == "daily":
        return dt.strftime("%b %d")
    if range_key == "weekly":
        return dt.strftime("Week of %b %d")
    if range_key == "monthly":
        return dt.strftime("%b %Y")
    raise ValueError(f"Unsupported range key: {range_key}")


def _build_risk_trend_data(collections):
    now = _utc_now()
    configs = {
        "daily": 7,
        "weekly": 8,
        "monthly": 6,
    }

    oldest_needed = _bucket_start(now, "monthly")
    for _ in range(configs["monthly"] - 1):
        oldest_needed = _bucket_start(oldest_needed - timedelta(days=1), "monthly")

    transactions_cursor = (
        collections["transactions"]
        .find()
        .sort("ingested_at", 1)
        .limit(5000)
    )
    alerts_cursor = (
        collections["alerts"]
        .find()
        .sort("created_at", 1)
        .limit(5000)
    )

    transaction_times = []
    for transaction in transactions_cursor:
        dt = _coerce_datetime(transaction.get("ingested_at") or transaction.get("timestamp"))
        if dt and dt >= oldest_needed:
            transaction_times.append(dt)

    alert_times = []
    for alert in alerts_cursor:
        dt = _coerce_datetime(alert.get("created_at"))
        if dt and dt >= oldest_needed:
            alert_times.append(dt)

    output = {}
    for range_key, bucket_count in configs.items():
        current_start = _bucket_start(now, range_key)
        starts = [current_start]
        while len(starts) < bucket_count:
            previous_anchor = starts[0] - timedelta(days=1)
            starts.insert(0, _bucket_start(previous_anchor, range_key))

        labels = [_format_bucket_label(start, range_key) for start in starts]
        tx_counts = []
        alert_counts = []
        risk_scores = []

        for start in starts:
            end = _next_bucket_start(start, range_key)
            tx_count = sum(1 for dt in transaction_times if start <= dt < end)
            alert_count = sum(1 for dt in alert_times if start <= dt < end)
            score = min(int((alert_count / tx_count) * 100), 100) if tx_count else 0

            tx_counts.append(tx_count)
            alert_counts.append(alert_count)
            risk_scores.append(score)

        output[range_key] = {
            "labels": labels,
            "risk_scores": risk_scores,
            "transaction_counts": tx_counts,
            "alert_counts": alert_counts,
        }

    return output


def _get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from openai import AzureOpenAI
        _embedding_model = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version="2023-05-15",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    return _embedding_model


def _search_azure_regulations(query: str):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery

    client = _get_embedding_model()
    response = client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    query_vector = response.data[0].embedding
    
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,
        fields="contentVector",
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "title", "content", "category", "jurisdiction"],
        top=3,
    )

    return [
        {
            "citation": result.get("title"),
            "document_id": result.get("id"),
            "category": result.get("category"),
            "jurisdiction": result.get("jurisdiction"),
            "snippet": result.get("content", "")[:900],
        }
        for result in results
    ]


def _search_local_regulations(query: str):
    if not LOCAL_REGULATORY_DOCS.exists():
        raise FileNotFoundError(
            "Local regulatory embeddings not found. Run "
            "`python -m agents.setup_azure_search` first."
        )

    with LOCAL_REGULATORY_DOCS.open("r", encoding="utf-8") as handle:
        documents = json.load(handle)

    client = _get_embedding_model()
    response = client.embeddings.create(input=query, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    query_vector = np.array(response.data[0].embedding)
    
    scored_docs = []
    for document in documents:
        doc_vector = np.array(document["contentVector"])
        similarity = np.dot(query_vector, doc_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        )
        scored_docs.append((similarity, document))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "citation": document.get("title"),
            "document_id": document.get("id"),
            "category": document.get("category"),
            "jurisdiction": document.get("jurisdiction"),
            "score": float(score),
            "snippet": document.get("content", "")[:900],
        }
        for score, document in scored_docs[:3]
    ]


def _search_regulations(query: str):
    if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
        return _search_azure_regulations(query)
    return _search_local_regulations(query)


@api_v1.route("/ingest", methods=["POST"])
def ingest_transactions():
    try:
        payload, error = _request_json()
        if error:
            return error

        records = _normalize_records(payload)
        now = _utc_now()
        documents = [
            {
                **record,
                "ingested_at": now,
                "ingest_source": record.get("ingest_source", "api"),
            }
            for record in records
        ]

        collection = get_collections()["transactions"]
        if len(documents) == 1:
            result = collection.insert_one(documents[0])
            inserted_ids = [str(result.inserted_id)]
        else:
            result = collection.insert_many(documents)
            inserted_ids = [str(inserted_id) for inserted_id in result.inserted_ids]

        logging.info("Ingested %s transaction(s)", len(inserted_ids))
        return jsonify({"status": "success", "inserted_ids": inserted_ids}), 201

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Failed to ingest transaction payload")
        return jsonify({"error": "Failed to ingest transactions", "details": str(exc)}), 500


@api_v1.route("/predict", methods=["POST"])
def predict_transactions():
    try:
        payload, error = _request_json()
        if error:
            return error

        records = _normalize_records(payload)
        artifacts = _load_model_artifacts()

        # Load user-configurable settings
        settings = _get_user_settings()
        ml_settings = settings.get("ml_engine", {})
        rf_weight = ml_settings.get("ensemble_rf_weight", 0.7)
        threshold = ml_settings.get("anomaly_threshold", 0.52)
        high_risk_regions = ml_settings.get("high_risk_regions", DEFAULT_HIGH_RISK_REGIONS)
        alert_amount_threshold = settings.get("general", {}).get("alert_amount_threshold", 0)

        features = _prepare_feature_matrix(records, high_risk_regions)

        if_model = artifacts["if_model"]
        rf_model = artifacts.get("rf_model")

        # IF scores (lower = more anomalous)
        if_raw_scores = if_model.decision_function(features)
        # Normalize IF scores to [0, 1] (inverted: higher = more anomalous)
        if_min, if_max = if_raw_scores.min(), if_raw_scores.max()
        if if_max - if_min > 0:
            if_anomaly_scores = 1 - (if_raw_scores - if_min) / (if_max - if_min)
        else:
            if_anomaly_scores = np.zeros_like(if_raw_scores)

        # Ensemble scoring
        if rf_model is not None:
            rf_probs = rf_model.predict_proba(features)[:, 1]
            ensemble_scores = rf_weight * rf_probs + (1 - rf_weight) * if_anomaly_scores
        else:
            # Fallback: IF-only mode
            ensemble_scores = if_anomaly_scores

        alerts_collection = get_collections()["alerts"]
        results = []

        for i, record in enumerate(records):
            transaction_id = record.get("transaction_id") or str(uuid.uuid4())
            score = float(ensemble_scores[i])
            is_anomaly = score >= threshold
            amount = float(record.get("amount", 0))

            # Only create alerts above the amount threshold
            if alert_amount_threshold > 0 and amount < alert_amount_threshold:
                is_anomaly = False

            response = {
                "transaction_id": transaction_id,
                "ensemble_score": round(score, 4),
                "if_anomaly_score": round(float(if_anomaly_scores[i]), 4),
                "rf_probability": round(float(rf_probs[i]), 4) if rf_model is not None else None,
                "is_anomaly": is_anomaly,
                "threshold": threshold,
            }

            if is_anomaly:
                alert = _build_alert(
                    {**record, "transaction_id": transaction_id},
                    float(if_raw_scores[i]),
                    score,
                )
                insert_result = alerts_collection.insert_one(alert)
                response["alert_id"] = str(insert_result.inserted_id)

            results.append(response)

        logging.info("Scored %s transaction(s) via ensemble", len(results))
        body = {"status": "success", "results": results}
        if len(results) == 1:
            body["result"] = results[0]
        return jsonify(body), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Failed to score transaction payload")
        return jsonify({"error": "Failed to score transactions", "details": str(exc)}), 500


@api_v1.route("/model_info", methods=["GET"])
def model_info():
    """Return metadata about the currently loaded ML model artifacts."""
    try:
        artifacts = _load_model_artifacts()
        config = artifacts.get("feature_config", {})
        feature_cols = artifacts.get("feature_columns", [])
        threshold = artifacts.get("threshold", "N/A")

        # Model artifact file sizes
        artifact_files = {}
        for name in ["isolation_forest", "random_forest", "scaler", "encoder_columns", "threshold", "feature_config"]:
            path = MODEL_DIR / f"{name}.joblib"
            if path.exists():
                artifact_files[name] = round(path.stat().st_size / 1024, 1)

        # IF params
        if_model = artifacts.get("if_model")
        if_info = {}
        if if_model is not None:
            if_info = {
                "n_estimators": if_model.n_estimators,
                "contamination": float(if_model.contamination),
                "max_samples": if_model.max_samples,
                "max_features": if_model.max_features,
            }

        # RF params
        rf_model = artifacts.get("rf_model")
        rf_info = {}
        if rf_model is not None:
            rf_info = {
                "n_estimators": rf_model.n_estimators,
                "max_depth": rf_model.max_depth,
                "n_features": rf_model.n_features_in_,
                "class_weight": str(rf_model.class_weight),
            }

        return jsonify({
            "status": "success",
            "model_info": {
                "feature_count": len(feature_cols),
                "feature_names": feature_cols,
                "trained_threshold": float(threshold) if isinstance(threshold, (int, float)) else threshold,
                "isolation_forest": if_info,
                "random_forest": rf_info,
                "ensemble_available": rf_model is not None,
                "high_risk_regions": config.get("high_risk_regions", DEFAULT_HIGH_RISK_REGIONS),
                "ensemble_rf_weight": config.get("ensemble_rf_weight", 0.7),
                "artifact_sizes_kb": artifact_files,
            },
        }), 200

    except Exception as exc:
        logging.exception("Failed to load model info")
        return jsonify({"error": "Failed to load model info", "details": str(exc)}), 500


@api_v1.route("/dashboard_live_data", methods=["GET"])
def dashboard_live_data():
    try:
        return jsonify(_get_dashboard_snapshot()), 200
    except Exception as exc:
        logging.exception("Failed to fetch dashboard live data")
        return jsonify({"error": "Failed to fetch dashboard live data", "details": str(exc)}), 500


@api_v1.route("/settings", methods=["GET", "POST"])
def user_settings():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Authentication required."}), 401

    try:
        collections = get_collections()
        users = collections["users"]

        if request.method == "GET":
            user = users.find_one({"_id": ObjectId(user_id)}, {"settings": 1})
            settings = _merge_settings((user or {}).get("settings"))
            return jsonify({"status": "success", "settings": settings}), 200

        payload, error = _request_json()
        if error:
            return error

        settings_payload = payload.get("settings") if isinstance(payload, dict) else None
        if not isinstance(settings_payload, dict):
            return jsonify({"error": "Payload must include a 'settings' object."}), 400

        settings = _merge_settings(settings_payload)
        users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"settings": settings, "updated_at": _utc_now()}},
            upsert=False,
        )
        return jsonify({"status": "success", "settings": settings}), 200

    except InvalidId:
        return jsonify({"error": "Invalid session user."}), 400
    except Exception as exc:
        logging.exception("Failed to load or save settings")
        return jsonify({"error": "Failed to load or save settings", "details": str(exc)}), 500


@api_v1.route("/search", methods=["POST"])
def search_regulations():
    try:
        payload, error = _request_json()
        if error:
            return error

        query = payload.get("query") if isinstance(payload, dict) else None
        if not query or not isinstance(query, str):
            return jsonify({"error": "Payload must include a non-empty 'query' string."}), 400

        results = _search_regulations(query)
        logging.info("Regulatory search completed for query: %s", query[:120])
        return jsonify({"status": "success", "query": query, "results": results}), 200

    except Exception as exc:
        logging.exception("Failed to search regulatory knowledge base")
        return jsonify({"error": "Failed to search regulatory context", "details": str(exc)}), 500


@api_v1.route("/chat", methods=["POST"])
def chat_compliance():
    """
    Conversational RAG endpoint for the Compliance Chatbot.
    Accepts: { "message": "...", "history": [ {"role": "user"|"assistant", "content": "..."}, ... ] }
    Returns: { "reply": "...", "sources": [...] }
    """
    try:
        payload, error = _request_json()
        if error:
            return error

        message = payload.get("message", "").strip() if isinstance(payload, dict) else ""
        if not message:
            return jsonify({"error": "Payload must include a non-empty 'message' string."}), 400

        history = payload.get("history", [])
        if not isinstance(history, list):
            history = []

        # ── Step 1: Retrieve RAG context ────────────────────────────
        rag_results = _search_regulations(message)
        context_blocks = []
        sources = []
        for doc in rag_results:
            citation = doc.get("citation", "Untitled")
            snippet = doc.get("snippet", "")
            category = doc.get("category", "General")
            jurisdiction = doc.get("jurisdiction", "")
            context_blocks.append(
                f"[{citation}] ({category} — {jurisdiction})\n{snippet}"
            )
            sources.append({
                "citation": citation,
                "category": category,
                "jurisdiction": jurisdiction,
                "document_id": doc.get("document_id", ""),
                "score": doc.get("score"),
            })

        rag_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No regulatory documents found."

        # ── Step 2: Build the prompt for Groq ───────────────────────
        system_prompt = (
            "You are the Veritas Compliance Assistant, an expert AI advisor on financial regulations, "
            "compliance frameworks, and risk management. You have access to a curated regulatory "
            "knowledge base that includes SEC regulations, Basel III, Bank Secrecy Act, OFAC sanctions, "
            "and SAR filing requirements.\n\n"
            "REGULATORY CONTEXT (retrieved from the knowledge base for this query):\n"
            f"{rag_context}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the user's question thoroughly using the regulatory context above.\n"
            "- Cite specific statutes, rules, and regulation names when applicable.\n"
            "- If the context doesn't contain enough information, say so honestly.\n"
            "- Use clear, professional language suitable for compliance officers.\n"
            "- Format your response with markdown (bold, bullet points, headers) for readability.\n"
            "- Keep answers focused and concise but comprehensive.\n"
        )

        # Build messages array for Groq
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limit to last 10 exchanges to stay within token limits)
        for entry in history[-20:]:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        # ── Step 3: Call Groq LLM ───────────────────────────────────
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        groq_model = os.getenv("GROQ_MODEL", "groq/llama3-70b-8192")

        # Strip the "groq/" prefix for direct API calls
        model_name = groq_model.replace("groq/", "") if groq_model.startswith("groq/") else groq_model

        from groq import Groq
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )

        reply = completion.choices[0].message.content

        logging.info("Chat response generated for query: %s", message[:80])
        return jsonify({
            "status": "success",
            "reply": reply,
            "sources": sources,
        }), 200

    except Exception as exc:
        logging.exception("Chat endpoint failed")
        return jsonify({"error": "Failed to generate response", "details": str(exc)}), 500


@api_v1.route("/advise", methods=["POST"])
def advise_on_alert():
    try:
        payload, error = _request_json()
        if error:
            return error

        alert_id = payload.get("alert_id") if isinstance(payload, dict) else None
        if not alert_id:
            return jsonify({"error": "Payload must include 'alert_id'."}), 400

        try:
            alert_object_id = ObjectId(alert_id)
        except InvalidId:
            return jsonify({"error": "Invalid MongoDB alert_id."}), 400

        collections = get_collections()
        alert = collections["alerts"].find_one({"_id": alert_object_id})
        if not alert:
            return jsonify({"error": "Alert not found."}), 404

        transaction_data = {
            **alert.get("transaction", {}),
            "alert_id": alert_id,
            "anomaly_score": alert.get("anomaly_score"),
            "ml_prediction": alert.get("prediction"),
        }

        from agents.risk_committee import run_risk_committee
        import threading

        # CrewAI uses asyncio internally. Flask's sync gunicorn worker may have
        # a stale event loop. We run the agent in a fresh thread with its own
        # event loop to avoid conflicts.
        result_holder = {"report": None, "error": None}

        def _run_agents():
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result_holder["report"] = run_risk_committee(transaction_data)
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        worker = threading.Thread(target=_run_agents)
        worker.start()
        worker.join(timeout=180)  # 3-minute timeout

        if result_holder["error"]:
            raise RuntimeError(result_holder["error"])
        if result_holder["report"] is None:
            raise RuntimeError("Agent execution timed out after 180 seconds")

        report = result_holder["report"]
        now = _utc_now()

        sar_doc = {
            "report": report,
            "created_at": now,
            "status": "draft",
        }
        
        agent_log = {
            "agent_system": "virtual_risk_committee",
            "model": os.getenv("GROQ_MODEL", "groq/compound"),
            "output": report,
            "created_at": now,
        }

        collections["alerts"].update_one(
            {"_id": alert_object_id},
            {
                "$set": {
                    "status": "advised",
                    "executive_brief": report,
                    "sar_draft": report,
                    "agent_log": agent_log,
                    "updated_at": now,
                }
            },
        )
        
        updated_alert = collections["alerts"].find_one({"_id": alert_object_id})

        logging.info("Generated advisory report for alert %s", alert_id)
        return jsonify(
            {
                "status": "success",
                "alert_id": alert_id,
                "report": report,
                "updated_alert": _jsonify_value(updated_alert)
            }
        ), 200

    except Exception as exc:
        logging.exception("Failed to run advisory workflow")
        return jsonify({"error": "Failed to run advisory workflow", "details": str(exc)}), 500


@api_v1.route("/portfolio/analyze", methods=["POST"])
def analyze_portfolio():
    """
    RAG/LLM endpoint that analyzes a client's investment portfolio
    and returns a strategic wealth advisory assessment.
    """
    try:
        from groq import Groq
        import pandas as pd
        
        payload = request.get_json() or {}
        client_id = payload.get("client_id")
        client_name = payload.get("client_name", "Unknown Client")
        
        if not client_id:
            return jsonify({"error": "No client_id provided"}), 400
            
        # Load rich data from CSV
        csv_path = str(PROJECT_ROOT / "data" / "raw" / "portfolios.csv")
        try:
            df = pd.read_csv(csv_path)
            client_df = df[df["Client_ID"] == client_id]
            if client_df.empty:
                return jsonify({"error": "Client data not found in CSV"}), 404
            
            # Select relevant rich fields to send to the LLM
            holdings = client_df[["Asset_Class", "Ticker", "Sector", "Invested_Amount", "Return_Pct", "Risk_Beta", "ESG_Score"]].to_dict(orient="records")
        except Exception as e:
            logging.error(f"Failed to read portfolio CSV: {e}")
            return jsonify({"error": "Data access error"}), 500
            
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Build prompt
        holdings_str = json.dumps(holdings, indent=2)
        sys_prompt = (
            "You are a Senior Fiduciary Wealth Advisor at Veritas Financial. "
            "Analyze the client's rich portfolio data (including Sector, Invested Amount, Return %, Risk Beta, and ESG Score). "
            "Provide exactly 3 bullet points of strategic advice regarding risk exposure, sector diversification, "
            "and portfolio volatility/ESG considerations. "
            "Do not include greetings or disclaimers, just the 3 bullet points formatted nicely in markdown."
        )
        user_prompt = f"Client: {client_name}\nHoldings:\n{holdings_str}\n\nPlease provide your expert analysis."
        
        model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        if model_name.startswith("groq/"):
            model_name = model_name[5:]
            
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        reply = completion.choices[0].message.content
        return jsonify({"status": "success", "analysis": reply})
        
    except Exception as exc:
        logging.exception("Failed to analyze portfolio")
        return jsonify({"error": "Failed to analyze portfolio", "details": str(exc)}), 500


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-fallback-key")
    app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/veritas")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    init_db(app)
    app.register_blueprint(api_v1)

    # ── API health check ───────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify(
            {
                "status": "Veritas API running",
                "version": "0.1.0",
                "timestamp": _utc_now().isoformat(),
            }
        ), 200

    @app.route("/api/debug/agents", methods=["GET"])
    def debug_agents():
        """Diagnostic endpoint: checks all agent dependencies."""
        checks = {}
        try:
            import crewai
            checks["crewai"] = f"OK (v{crewai.__version__})"
        except Exception as e:
            checks["crewai"] = f"FAIL: {e}"
        try:
            from crewai import LLM
            checks["crewai_llm"] = "OK"
        except Exception as e:
            checks["crewai_llm"] = f"FAIL: {e}"
        try:
            from groq import Groq
            checks["groq"] = "OK"
        except Exception as e:
            checks["groq"] = f"FAIL: {e}"
        try:
            from agents.risk_committee import run_risk_committee
            checks["risk_committee_import"] = "OK"
        except Exception as e:
            checks["risk_committee_import"] = f"FAIL: {e}"

        checks["GROQ_API_KEY"] = "SET" if os.getenv("GROQ_API_KEY") else "MISSING"
        checks["GROQ_MODEL"] = os.getenv("GROQ_MODEL", "NOT SET")
        checks["AZURE_SEARCH_ENDPOINT"] = "SET" if os.getenv("AZURE_SEARCH_ENDPOINT") else "MISSING"

        return jsonify(checks), 200

    # ── Auth decorator ──────────────────────────────────────
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login_page'))
            return f(*args, **kwargs)
        return decorated_function

    # ── Frontend page routes ────────────────────────────────
    @app.route("/")
    @app.route("/landing")
    def landing_page():
        return render_template("landing.html")

    @app.route("/dashboard")
    @login_required
    def dashboard_page():
        snapshot = _get_dashboard_snapshot()

        return render_template(
            "dashboard.html",
            active_page="dashboard",
            metrics=snapshot["metrics"],
            recent_transactions=snapshot["recent_transactions"],
        )

    @app.route("/alerts")
    @login_required
    def alerts_page():
        collections = get_collections()
        # Fetch all alerts, sorted by _id descending (CosmosDB-safe)
        alerts_cursor = (
            collections["alerts"]
            .find()
            .sort("_id", -1)
        )
        alerts_data = [_jsonify_value(doc) for doc in alerts_cursor]
        return render_template(
            "alerts.html",
            active_page="alerts",
            alerts_data=alerts_data,
        )

    @app.route("/settings")
    @login_required
    def settings_page():
        return render_template("settings.html", active_page="settings")

    @app.route("/portfolio")
    @login_required
    def portfolio_page():
        return render_template("portfolio.html", active_page="portfolio")

    @app.route("/compliance")
    @login_required
    def compliance_page():
        return render_template("compliance.html", active_page="compliance")

    @app.route("/investigate/<transaction_id>")
    @login_required
    def investigate_page(transaction_id):
        collections = get_collections()

        # Find the transaction document by its transaction_id field
        transaction = collections["transactions"].find_one(
            {"transaction_id": transaction_id}
        )
        if not transaction:
            return render_template(
                "base.html",
                active_page="investigation",
                error="Transaction not found.",
            ), 404

        # Look for a linked alert (ML-flagged)
        alert = collections["alerts"].find_one(
            {"transaction_id": transaction_id}
        )

        return render_template(
            "investigate.html",
            active_page="investigation",
            transaction=_jsonify_value(transaction),
            alert=_jsonify_value(alert) if alert else None,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login_page():
        if request.method == "GET":
            return render_template("login.html")

        # ── POST: authenticate user ─────────────────────────
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "error")
            return render_template("login.html"), 400

        collections = get_collections()
        try:
            user = collections["users"].find_one({"email": email})
        except Exception as e:
            logging.error("Login query failed: %s", e)
            flash("Service temporarily unavailable. Please try again.", "error")
            return render_template("login.html"), 503

        if not user or not check_password_hash(user.get("password_hash", ""), password):
            flash("Invalid credentials. Please try again.", "error")
            return render_template("login.html"), 401

        # Success — create session
        session["user_id"] = str(user["_id"])
        session["user_name"] = user.get("name", "User")
        session["user_role"] = user.get("role", "Analyst")
        logging.info("User %s logged in successfully.", email)
        return redirect(url_for("dashboard_page"))

    @app.route("/register", methods=["GET", "POST"])
    def register_page():
        if request.method == "GET":
            return render_template("register.html")

        # ── POST: create new user ───────────────────────────
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "Risk Analyst").strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html"), 400

        collections = get_collections()

        # Check for duplicate email
        if collections["users"].find_one({"email": email}):
            flash("An account with this email already exists.", "error")
            return render_template("register.html"), 409

        # Insert new user
        user_doc = {
            "name": name,
            "email": email,
            "role": role,
            "password_hash": generate_password_hash(password),
            "created_at": _utc_now(),
        }
        try:
            collections["users"].insert_one(user_doc)
        except Exception as e:
            logging.error("Failed to create user: %s", e)
            flash("Account creation failed. Please try again later.", "error")
            return render_template("register.html"), 500

        logging.info("New user registered: %s (%s)", email, role)
        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for("login_page"))

    @app.route("/logout")
    def logout_page():
        session.clear()
        return redirect(url_for("landing_page"))

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )

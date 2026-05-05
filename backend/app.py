"""
Veritas Financial Risk & Advisory Platform
Flask Backend API

Endpoints:
  GET  /                         Health check
  POST /api/v1/ingest            Persist raw transactions
  POST /api/v1/predict           Score transactions with IsolationForest
  POST /api/v1/search            Retrieve regulatory context
  POST /api/v1/advise            Run the Virtual Risk Committee
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import Blueprint, Flask, jsonify, request

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")

CATEGORICAL_FEATURES = ["transaction_type", "currency", "region"]
TEMPORAL_FEATURES = ["hour", "day_of_week", "is_weekend", "is_night"]

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
        "model": MODEL_DIR / "isolation_forest.joblib",
        "scaler": MODEL_DIR / "scaler.joblib",
        "feature_columns": MODEL_DIR / "encoder_columns.joblib",
        "feature_config": MODEL_DIR / "feature_config.joblib",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {missing}")

    _model_cache.update(
        {
            name: joblib.load(path)
            for name, path in required_files.items()
        }
    )
    return _model_cache


def _prepare_feature_matrix(records):
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

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    ts = ts.fillna(pd.Timestamp(_utc_now()))

    df = df.assign(
        hour=ts.dt.hour,
        day_of_week=ts.dt.dayofweek,
        is_weekend=(ts.dt.dayofweek >= 5).astype(int),
        is_night=((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int),
        amount_log=np.log1p(df["amount"]),
    )

    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    feature_columns = _load_model_artifacts()["feature_columns"]
    features = encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = _load_model_artifacts()["scaler"].transform(features)

    return scaled


def _build_alert(record, prediction, anomaly_score):
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
        "prediction": int(prediction),
        "anomaly_score": float(anomaly_score),
        "status": "open",
        "created_at": _utc_now(),
        "source": "ml_isolation_forest",
    }


def _get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def _search_azure_regulations(query: str):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery

    query_vector = _get_embedding_model().encode(query).tolist()
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

    query_vector = np.array(_get_embedding_model().encode(query))
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
        features = _prepare_feature_matrix(records)

        model = artifacts["model"]
        predictions = model.predict(features)
        anomaly_scores = model.decision_function(features)

        alerts_collection = get_collections()["alerts"]
        results = []

        for record, prediction, anomaly_score in zip(records, predictions, anomaly_scores):
            transaction_id = record.get("transaction_id") or str(uuid.uuid4())
            response = {
                "transaction_id": transaction_id,
                "prediction": int(prediction),
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(prediction == -1),
            }

            if prediction == -1:
                alert = _build_alert({**record, "transaction_id": transaction_id}, prediction, anomaly_score)
                insert_result = alerts_collection.insert_one(alert)
                response["alert_id"] = str(insert_result.inserted_id)

            results.append(response)

        logging.info("Scored %s transaction(s)", len(results))
        body = {"status": "success", "results": results}
        if len(results) == 1:
            body["result"] = results[0]
        return jsonify(body), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Failed to score transaction payload")
        return jsonify({"error": "Failed to score transactions", "details": str(exc)}), 500


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
        import concurrent.futures

        # Run CrewAI in a completely isolated process to avoid Flask threading/asyncio conflicts
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_risk_committee, transaction_data)
            report = future.result()
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

    @app.route("/", methods=["GET"])
    def health_check():
        return jsonify(
            {
                "status": "Veritas API running",
                "version": "0.1.0",
                "timestamp": _utc_now().isoformat(),
            }
        ), 200

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )

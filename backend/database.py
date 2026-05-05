"""
MongoDB initialization for the Veritas Flask backend.

The module exposes a single Flask-PyMongo extension instance plus collection
references that are bound when ``init_db(app)`` is called by the application
factory.
"""

import os

from dotenv import load_dotenv
from flask_pymongo import PyMongo

load_dotenv()

mongo = PyMongo()

transactions = None
alerts = None
sar_documents = None
agent_logs = None


def init_db(app):
    """Initialize PyMongo and bind the collection references."""

    app.config.setdefault(
        "MONGO_URI",
        os.getenv("MONGO_URI", "mongodb://localhost:27017/veritas"),
    )
    mongo.init_app(app)
    bind_collections()
    return mongo


def bind_collections():
    """Bind module-level collection handles after PyMongo is initialized."""

    global transactions, alerts, sar_documents, agent_logs

    database = mongo.db
    if database is None:
        database_name = os.getenv("MONGO_DB_NAME", "veritas")
        database = mongo.cx[database_name]

    transactions = database.transactions
    alerts = database.alerts
    sar_documents = database.sar_documents
    agent_logs = database.agent_logs


def get_collections():
    """Return the configured MongoDB collections as a small registry."""

    if any(
        collection is None
        for collection in (transactions, alerts, sar_documents, agent_logs)
    ):
        bind_collections()

    return {
        "transactions": transactions,
        "alerts": alerts,
        "sar_documents": sar_documents,
        "agent_logs": agent_logs,
    }

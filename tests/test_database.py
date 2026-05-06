import pytest
from unittest.mock import patch, MagicMock

def test_database_init_db():
    with patch('pymongo.MongoClient') as mock_client:
        from backend.database import init_db
        mock_app = MagicMock()
        mock_app.config = {"MONGO_URI": "mongodb://localhost:27017/test"}
        db = init_db(mock_app)
        assert db is not None

def test_get_collections():
    with patch('backend.database.mongo') as mock_mongo:
        mock_db = MagicMock()
        mock_mongo.db = mock_db
        mock_db.transactions = MagicMock()
        mock_db.alerts = MagicMock()
        mock_db.users = MagicMock()
        from backend.database import get_collections
        cols = get_collections()
        assert "transactions" in cols
        assert "alerts" in cols
        assert "users" in cols

import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def app_instance():
    # Mock Cosmos DB connection before importing app
    import backend.database as database

    with patch.object(database, 'init_db'), patch.object(database, 'get_collections') as mock_get_colls:
        mock_tx = MagicMock()
        mock_tx.count_documents.return_value = 50
        mock_tx.find.return_value.sort.return_value.limit.return_value = []
        mock_alerts = MagicMock()
        mock_alerts.count_documents.return_value = 10
        mock_alerts.find.return_value.sort.return_value.limit.return_value = []
        mock_users = MagicMock()
        mock_users.find_one.return_value = {"settings": {"ml_engine": {"contamination_rate": 0.08}}}
        mock_colls = {"transactions": mock_tx, "alerts": mock_alerts, "users": mock_users}
        mock_get_colls.return_value = mock_colls
        
        from backend.app import create_app
        app = create_app()
        app.config['TESTING'] = True
        return app

@pytest.fixture
def client(app_instance):
    """Create a Flask test client."""
    with app_instance.test_client() as client:
        yield client

def test_dashboard_route_loads(client):
    """Test that the dashboard HTML loads successfully."""
    # Since DB is mocked, we need to mock session for login
    with client.session_transaction() as sess:
        sess['user_id'] = 'mock_user'
        sess['user_name'] = 'Mock User'
        
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"metric-transactions" in response.data

def test_dashboard_live_data_endpoint(client):
    """Test that dashboard live data returns metrics and recent transactions."""
    response = client.get('/api/v1/dashboard_live_data')
    assert response.status_code == 200

    data = response.get_json()
    assert "metrics" in data
    assert "recent_transactions" in data
    assert "chart_data" in data
    assert data["metrics"]["total_transactions"] == 50
    assert data["metrics"]["total_alerts"] == 10
    assert set(data["chart_data"].keys()) == {"daily", "weekly", "monthly"}

def test_api_predict_missing_payload(client):
    """Test /api/v1/predict with no payload."""
    response = client.post('/api/v1/predict')
    # Should fail because of missing content-type/JSON
    assert response.status_code == 400

def test_settings_load_and_save(client, mocker):
    mock_users = MagicMock()
    mock_users.find_one.return_value = {"settings": {"agents": {"cro": False}}}
    mocker.patch('backend.app.get_collections', return_value={
        "transactions": MagicMock(),
        "alerts": MagicMock(),
        "users": mock_users,
    })

    with client.session_transaction() as sess:
        sess['user_id'] = '507f1f77bcf86cd799439011'

    load_response = client.get('/api/v1/settings')
    assert load_response.status_code == 200
    load_data = load_response.get_json()
    assert load_data["settings"]["agents"]["cro"] is False
    assert load_data["settings"]["general"]["platform_name"] == "Veritas Financial Advisory"

    save_response = client.post('/api/v1/settings', json={
        "settings": {
            "general": {"platform_name": "Custom Veritas"},
            "agents": {"compliance": False}
        }
    })
    assert save_response.status_code == 200
    mock_users.update_one.assert_called_once()

def test_api_predict_valid_payload(client, mocker):
    """Test /api/v1/predict with a valid transaction."""
    payload = {
        "transaction_id": "txn_test_123",
        "amount": 500,
        "region": "North America",
        "account_metadata": {
            "account_holder": "Test User",
            "historical_avg_transaction": 450
        }
    }
    
    # Mock ML scoring internal methods
    mocker.patch('backend.app._load_model_artifacts', return_value={
        "model": MagicMock(),
        "scaler": MagicMock(),
        "categorical_cols": [],
        "metadata": {}
    })
    mocker.patch('backend.app._prepare_feature_matrix', return_value=(MagicMock(), MagicMock()))
    mocker.patch('backend.app._build_alert', return_value={"mock": "alert"})
    
    # Mock DB collections to avoid real DB interaction during predict
    mocker.patch('backend.app.get_collections', return_value={
        "transactions": MagicMock(),
        "alerts": MagicMock(),
        "users": MagicMock()
    })
    
    response = client.post('/api/v1/predict', json=payload)
    
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    
def test_api_chat_endpoint(client, mocker):
    """Test the /api/v1/chat RAG endpoint."""
    payload = {
        "message": "What is the Bank Secrecy Act?",
        "history": []
    }
    
    # Mock RAG search to return fake context
    mocker.patch('backend.app._search_regulations', return_value=[
        {
            "citation": "BSA Sec 101",
            "snippet": "Financial institutions must keep records of cash purchases...",
            "category": "AML",
            "jurisdiction": "USA"
        }
    ])
    
    # Mock Groq LLM response using the actual module where Groq is defined
    mock_groq = mocker.patch('groq.Groq')
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "The Bank Secrecy Act requires banks to keep records..."
    mock_groq.return_value.chat.completions.create.return_value = mock_completion
    
    response = client.post('/api/v1/chat', json=payload)
    
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "reply" in data
    assert "The Bank Secrecy Act requires" in data["reply"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["citation"] == "BSA Sec 101"

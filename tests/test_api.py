import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def app_instance():
    # Mock Cosmos DB connection before importing app
    with patch('backend.database.init_db'), patch('backend.database.get_collections') as mock_get_colls:
        mock_tx = MagicMock()
        mock_tx.count_documents.return_value = 50
        mock_alerts = MagicMock()
        mock_alerts.count_documents.return_value = 10
        mock_colls = {"transactions": mock_tx, "alerts": mock_alerts, "users": MagicMock()}
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
    assert b"Risk &amp; Portfolio Overview" in response.data

def test_api_predict_missing_payload(client):
    """Test /api/v1/predict with no payload."""
    response = client.post('/api/v1/predict')
    # Should fail because of missing content-type/JSON
    assert response.status_code == 400

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

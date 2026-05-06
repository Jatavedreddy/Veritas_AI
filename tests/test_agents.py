import pytest
from unittest.mock import patch, MagicMock

@patch('crewai.Crew')
def test_run_risk_committee_success(mock_crew_class):
    """Test that the risk committee successfully returns a SAR string."""
    from agents.risk_committee import run_risk_committee
    
    # Mock the crew execution to return a simple string
    mock_crew_instance = mock_crew_class.return_value
    mock_crew_instance.kickoff.return_value = "SAR REPORT: Suspicious Activity Detected."
    
    transaction_data = {
        "transaction_id": "mock_txn_123",
        "amount": 500000,
        "currency": "USD",
        "region": "High-Risk Jurisdiction",
        "anomaly_score": -0.8
    }
    
    result = run_risk_committee(transaction_data)
    
    assert "SAR REPORT:" in result
    mock_crew_instance.kickoff.assert_called_once()

@patch('crewai.Crew')
def test_run_risk_committee_handles_exception(mock_crew_class):
    """Test that the risk committee gracefully handles errors."""
    from agents.risk_committee import run_risk_committee
    
    mock_crew_instance = mock_crew_class.return_value
    mock_crew_instance.kickoff.side_effect = Exception("Groq API Timeout")
    
    transaction_data = {
        "transaction_id": "mock_txn_123",
        "amount": 500000,
        "currency": "USD",
        "region": "High-Risk Jurisdiction",
        "anomaly_score": -0.8
    }
    
    with pytest.raises(Exception) as excinfo:
        run_risk_committee(transaction_data)
    
    assert "Groq API Timeout" in str(excinfo.value)

import pytest
import numpy as np
from ml.train_model import train_isolation_forest

def test_train_isolation_forest():
    """Test that train_isolation_forest returns a fitted model."""
    # Create fake scaled data (100 samples, 5 features)
    np.random.seed(42)
    fake_X_scaled = np.random.rand(100, 5)
    
    model = train_isolation_forest(fake_X_scaled)
    
    assert model is not None
    # Verify the model can make predictions
    preds = model.predict(fake_X_scaled)
    assert len(preds) == 100
    
    # IsolationForest returns 1 for inliers, -1 for outliers
    assert all(p in (1, -1) for p in preds)

def test_isolation_forest_scores():
    """Test that the isolation forest can output anomaly scores."""
    np.random.seed(42)
    fake_X_scaled = np.random.rand(100, 5)
    model = train_isolation_forest(fake_X_scaled)
    
    scores = model.decision_function(fake_X_scaled)
    assert len(scores) == 100
    assert isinstance(scores[0], float)

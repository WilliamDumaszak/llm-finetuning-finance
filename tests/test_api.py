import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    from api.main import app, MODEL_STATE
    # Inject a mock model so tests don't require actual model weights
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}

    MODEL_STATE["model"] = mock_model
    MODEL_STATE["tokenizer"] = mock_tokenizer
    MODEL_STATE["device"] = "cpu"
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_no_model():
    from api.main import app, MODEL_STATE
    MODEL_STATE.clear()
    c = TestClient(app)
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_no_model():
    from api.main import app, MODEL_STATE
    MODEL_STATE.clear()
    c = TestClient(app)
    response = c.post("/predict", json={"text": "The earnings were strong this quarter."})
    assert response.status_code == 503


def test_predict_text_too_short(client):
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422

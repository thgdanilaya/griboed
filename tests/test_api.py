import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import app  # noqa: E402

client = TestClient(app)


def test_root_lists_service_endpoints() -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["predict_endpoint"] == "/predict"
    assert payload["monitoring_endpoint"] == "/monitoring"
    assert payload["ui_endpoint"] == "/ui"


def test_ui_returns_html_page() -> None:
    response = client.get("/ui")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Mushroom Classifier" in response.text


def test_monitoring_returns_service_and_model_state() -> None:
    response = client.get("/monitoring")

    assert response.status_code == 200
    payload = response.json()
    assert "service" in payload
    assert "model" in payload
    assert "prediction_metrics" in payload
    assert "infrastructure" in payload
    assert "metrics" in payload
    assert isinstance(payload["model"]["loaded"], bool)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "roc-auc"} <= set(
        payload["metrics"]
    )

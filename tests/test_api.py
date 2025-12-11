import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "solution-guidance"))

import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Patch model functions to avoid heavy operations during API tests
    called = {"train": False, "predict": False}

    def fake_train(data_dir=None, test=False):
        called["train"] = True
        called["data_dir"] = data_dir
        called["test"] = test

    def fake_load(training=False):
        return {"all": {}}, {"all": object()}

    def fake_predict(country, year, month, day, **kwargs):
        called["predict"] = True
        called["predict_args"] = (country, year, month, day)
        return {"y_pred": [42.0], "y_proba": None}

    monkeypatch.setattr(app.model, "model_train", fake_train)
    monkeypatch.setattr(app.model, "model_load", fake_load)
    monkeypatch.setattr(app.model, "model_predict", fake_predict)

    with app.app.test_client() as client:
        yield client, called


def test_health(client):
    client_obj, _ = client
    response = client_obj.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_train_and_predict_endpoints(client):
    client_obj, called = client

    train_resp = client_obj.post("/train", json={"data_dir": "cs-train", "test": True})
    assert train_resp.status_code == 200
    assert called["train"] is True

    predict_resp = client_obj.post(
        "/predict", json={"country": "all", "year": "2019", "month": "01", "day": "10"}
    )
    assert predict_resp.status_code == 200
    assert called["predict"] is True
    assert predict_resp.get_json()["prediction"] == 42.0


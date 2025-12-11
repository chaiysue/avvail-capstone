import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "solution-guidance"))

import model  # noqa: E402


def _make_sample_data(tmp_path: Path, days: int = 120) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = datetime(2019, 1, 1)
    for i in range(days):
        current = start + timedelta(days=i)
        rows.append(
            {
                "country": "all",
                "customer_id": i,
                "day": current.day,
                "invoice": f"{1000+i}",
                "month": current.month,
                "price": float(100 + i % 10),
                "stream_id": i % 5,
                "times_viewed": int(1 + (i % 3)),
                "year": current.year,
            }
        )
    df = pd.DataFrame(rows)
    df.to_json(data_dir / "sample.json", orient="records")
    return data_dir


def test_model_train_and_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = _make_sample_data(tmp_path)

    model.MODEL_DIR = str(tmp_path / "models")

    # Train on synthetic data (test flag ensures lightweight artifacts)
    model.model_train(data_dir=str(data_dir), test=True)
    saved_models = list(Path(model.MODEL_DIR).glob("test-*.joblib"))
    assert saved_models, "Expected serialized model artifacts"

    # Load the test models and run a prediction
    all_data, all_models = model.model_load(prefix="test", data_dir=str(data_dir), training=False)
    result = model.model_predict(
        "all", "2019", "03", "15", all_models=all_models, all_data=all_data, test=True
    )
    assert "y_pred" in result
    assert result["y_pred"] is not None


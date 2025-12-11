import csv
import os
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "solution-guidance"))

from logger import update_predict_log, update_train_log  # noqa: E402


def test_update_train_and_predict_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"

    update_train_log(
        tag="all",
        date_range=("2020-01-01", "2020-02-01"),
        metrics={"rmse": 10.5},
        runtime="00:00:01",
        model_version=0.1,
        model_version_note="test",
        test=True,
    )
    update_predict_log(
        country="all",
        y_pred=[123.4],
        y_proba=None,
        target_date="2020-01-15",
        runtime="00:00:00",
        model_version=0.1,
        test=True,
    )

    train_log = log_dir / "train-test.log"
    predict_log = log_dir / "predict-test.log"
    assert train_log.exists()
    assert predict_log.exists()

    with train_log.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["tag"] == "all"
    assert rows[0]["rmse"] == "10.5"

    with predict_log.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["country"] == "all"
    assert float(rows[0]["prediction"]) == 123.4


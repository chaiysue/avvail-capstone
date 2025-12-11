"""
Lightweight logging helpers for training and prediction events.
Designed to keep test artifacts isolated from production logs.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

DEFAULT_LOG_DIR = "logs"


def _ensure_log_dir(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _write_log(log_path: str, headers: Iterable[str], row: Dict):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(headers))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_train_log(
    tag: str,
    date_range: Tuple[str, str],
    metrics: Dict[str, float],
    runtime: str,
    model_version: float,
    model_version_note: str,
    test: bool = False,
    log_dir: str = DEFAULT_LOG_DIR,
):
    """
    Append a training event to the train log. Uses a separate file when test=True.
    """
    _ensure_log_dir(log_dir)
    log_name = "train-test.log" if test else "train.log"
    log_path = os.path.join(log_dir, log_name)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "date_range": "{}|{}".format(*date_range) if date_range else "",
        "rmse": metrics.get("rmse") if metrics else None,
        "baseline_rmse": metrics.get("baseline_rmse") if metrics else None,
        "selected_model": metrics.get("selected_model") if metrics else None,
        "runtime": runtime,
        "model_version": model_version,
        "model_version_note": model_version_note,
        "test": test,
    }
    headers = row.keys()
    _write_log(log_path, headers, row)


def update_predict_log(
    country: str,
    y_pred,
    y_proba,
    target_date: str,
    runtime: str,
    model_version: float,
    test: bool = False,
    log_dir: str = DEFAULT_LOG_DIR,
):
    """
    Append a prediction event to the predict log. Uses a separate file when test=True.
    """
    _ensure_log_dir(log_dir)
    log_name = "predict-test.log" if test else "predict.log"
    log_path = os.path.join(log_dir, log_name)
    pred_value = None
    if y_pred is not None:
        pred_value = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)
    proba_value = None
    if y_proba is not None:
        proba_value = y_proba if isinstance(y_proba, (float, int)) else str(y_proba)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "target_date": target_date,
        "prediction": pred_value,
        "proba": proba_value,
        "runtime": runtime,
        "model_version": model_version,
        "test": test,
    }
    headers = row.keys()
    _write_log(log_path, headers, row)

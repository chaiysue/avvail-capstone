"""
Minimal monitoring utilities to surface model activity and basic sanity checks
from the train/predict logs.
"""

import os
import pandas as pd

LOG_DIR = "logs"


def _load_log(name: str, log_dir: str = LOG_DIR) -> pd.DataFrame:
    path = os.path.join(log_dir, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize(log_dir: str = LOG_DIR) -> dict:
    train_df = _load_log("train.log", log_dir)
    predict_df = _load_log("predict.log", log_dir)

    summary = {
        "train_events": len(train_df),
        "predict_events": len(predict_df),
    }
    if not train_df.empty:
        summary["last_train_model_version"] = train_df["model_version"].iloc[-1]
        summary["last_train_rmse"] = train_df["rmse"].iloc[-1]
    if not predict_df.empty:
        summary["prediction_mean"] = predict_df["prediction"].mean()
        summary["prediction_std"] = predict_df["prediction"].std()
    return summary


def detect_anomalies(log_dir: str = LOG_DIR, max_abs_prediction: float = 1e7) -> list:
    """
    Flag obviously bad predictions to trigger re-training or deeper diagnostics.
    """
    predict_df = _load_log("predict.log", log_dir)
    if predict_df.empty:
        return ["no_predictions"]
    issues = []
    if predict_df["prediction"].abs().max() > max_abs_prediction:
        issues.append("prediction_out_of_bounds")
    if predict_df["prediction"].std() == 0:
        issues.append("prediction_no_variation")
    return issues


if __name__ == "__main__":
    print("monitor summary:", summarize())
    print("monitor anomalies:", detect_anomalies())


import os
import sys
from flask import Flask, jsonify, request

# Ensure local modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOLUTION_PATH = os.path.join(PROJECT_ROOT, "solution-guidance")
if SOLUTION_PATH not in sys.path:
    sys.path.append(SOLUTION_PATH)

import model  # noqa: E402

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    data_dir = payload.get("data_dir", "cs-train")
    test_flag = bool(payload.get("test", False))
    try:
        model.model_train(data_dir=data_dir, test=test_flag)
        return (
            jsonify(
                {
                    "status": "trained",
                    "data_dir": data_dir,
                    "test": test_flag,
                }
            ),
            200,
        )
    except Exception as exc:  # pragma: no cover - surfaced via API response
        return jsonify({"error": str(exc)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "request body required"}), 400

    country = payload.get("country", "all")
    year = str(payload.get("year", "")).zfill(4)
    month = str(payload.get("month", "")).zfill(2)
    day = str(payload.get("day", "")).zfill(2)
    test_flag = bool(payload.get("test", False))
    try:
        all_data, all_models = model.model_load(training=False)
        result = model.model_predict(
            country,
            year,
            month,
            day,
            all_models=all_models,
            all_data=all_data,
            test=test_flag,
        )
        y_pred = result.get("y_pred")
        y_value = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)
        return (
            jsonify(
                {
                    "country": country,
                    "target_date": f"{year}-{month}-{day}",
                    "prediction": y_value,
                }
            ),
            200,
        )
    except Exception as exc:  # pragma: no cover - surfaced via API response
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)


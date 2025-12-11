# Repository Guidelines

## Project Structure & Module Organization
- `cs-train/`: monthly JSON invoice history for training/EDA; keep schema intact and treat as read-only source data.
- `cs-production/`: hold-out or production-shaped JSON with the same schema; do not edit in place.
- `solution-guidance/`: utilities; `cslib.py` handles ingestion/feature engineering, `model.py` trains/predicts via scikit-learn random forests, `logger.py` writes train/predict logs.
- Runtime artifacts: `solution-guidance/ts-data/` (cached CSVs from `fetch_ts`), `models/` (joblib artifacts), and `logs/` (CSV logs) are created when you train; avoid committing them unless required.

## Build, Test, and Development Commands
- Use Python 3.8+ with a virtual env: `python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt`.
- Make utilities importable: PowerShell example `$env:PYTHONPATH="$PWD/solution-guidance"` before using `model` or `cslib` interactively.
- Train quickly: `python -c "import model; model.model_train('cs-train', test=True)"` (creates/updates `models/` and logs).
- Predict from saved models: `python -c "import model; model.model_load(training=False); model.model_predict('all','2018','01','05')"`.
- Run all tests: `python run-tests.py` (pytest). API smoke: `uv run flask --app app run --port 8080` or `uv run gunicorn -b 0.0.0.0:8080 app:app`.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indents, snake_case for functions/vars, CapWords for classes; keep functions small and composable.
- Validate inputs early (see `fetch_data` and `convert_to_ts`) and prefer deterministic behavior (set seeds when adding randomness).
- Document new helpers with concise docstrings; keep path handling OS-agnostic via `os.path.join`.

## Testing Guidelines
- Pytest suite lives in `tests/` (`test_model.py`, `test_api.py`, `test_logger.py`). Use temp dirs/fixtures to isolate artifacts from production logs/models.
- For smoke tests, run `model.model_train('cs-train', test=True)` then `model.model_predict(...)` on a known date; confirm RMSE output and new artifacts under `models/`.
- When adding data sources, include schema checks and small synthetic fixtures instead of committing large blobs.

## Commit & Pull Request Guidelines
- With no existing git history, use concise, imperative messages (e.g., `feat: add ts feature checks`) and note model version bumps (`MODEL_VERSION`/`MODEL_VERSION_NOTE`).
- PRs should describe data used, metrics (e.g., RMSE), artifacts produced, and any new configuration paths; attach sample command output or screenshots when relevant.
- Avoid committing generated artifacts (`models/*.joblib`, `solution-guidance/ts-data/*.csv`, `logs/*.log`) unless explicitly requested; document regeneration steps instead.

## Security & Configuration Tips
- Do not embed credentials; data files are local and should remain read-only. Keep `logger.py` minimal and free of secrets.
- Keep path and file handling platform-agnostic; prefer relative paths rooted at the repo when sharing commands.

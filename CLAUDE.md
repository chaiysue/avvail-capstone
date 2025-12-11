# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the IBM AI Enterprise Workflow Capstone project - a time-series revenue forecasting system for AAVAIL. The system ingests JSON invoice data, trains per-country Random Forest/SVR models, and serves predictions via a Flask API. The project follows a 3-part structure: (1) data exploration and EDA, (2) model development and iteration, (3) API deployment with monitoring.

## Development Commands

### Environment Setup

**Using uv (Recommended):**
```bash
# Install dependencies and create virtual environment automatically
uv sync

# uv automatically manages the virtual environment, no activation needed
# All commands use: uv run <command>
```

**Using traditional Python venv:**
```bash
# Create and activate virtual environment (Python 3.8-3.11)
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt

# Make solution-guidance modules importable (PowerShell)
$env:PYTHONPATH="$PWD/solution-guidance"

# Unix/Mac
export PYTHONPATH="$PWD/solution-guidance"
```

**Note:** This project requires Python 3.8-3.11 (not 3.12+) due to numpy/pandas version constraints.

### Training and Prediction
```bash
# With uv (recommended)
# Quick training (test mode = subset data + fast execution)
uv run python -c "import sys; sys.path.append('solution-guidance'); import model; model.model_train('cs-train', test=True)"

# Full training (all countries, all data)
uv run python -c "import sys; sys.path.append('solution-guidance'); import model; model.model_train('cs-train', test=False)"

# Load models and predict
uv run python -c "import sys; sys.path.append('solution-guidance'); import model; all_data, all_models = model.model_load(prefix='test', training=False); model.model_predict('all','2018','01','05', all_models=all_models, all_data=all_data)"

# With traditional setup (requires PYTHONPATH set)
# Quick training
python -c "import model; model.model_train('cs-train', test=True)"

# Full training
python -c "import model; model.model_train('cs-train', test=False)"

# Predict
python -c "import model; model.model_load(training=False); model.model_predict('all','2018','01','05')"
```

### Testing
```bash
# With uv (recommended)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_model.py -v
uv run pytest tests/test_api.py -v

# With traditional setup
python run-tests.py
python -m pytest tests/test_model.py -v
```

### API Development
```bash
# With uv (recommended)
# Run Flask development server
uv run flask --app app run --port 8080

# Run with gunicorn (production-like)
uv run gunicorn -b 0.0.0.0:8080 app:app

# With traditional setup
flask --app app run --port 8080
gunicorn -b 0.0.0.0:8080 app:app

# Docker build and run (uses requirements.txt)
docker build -t avvail-api .
docker run -p 8080:8080 avvail-api
```

### API Usage
```bash
# Health check
curl http://localhost:8080/health

# Train models via API
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "cs-train", "test": true}'

# Get prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "all", "year": "2018", "month": "01", "day": "05"}'
```

### Monitoring
```bash
# With uv (recommended)
uv run python -c "import monitor; print(monitor.summarize())"
uv run python -c "import monitor; print(monitor.detect_anomalies())"

# With traditional setup
python -c "import monitor; print(monitor.summarize())"
python -c "import monitor; print(monitor.detect_anomalies())"
```

## Architecture Overview

### Project Files
- **`pyproject.toml`**: Modern Python project configuration for uv dependency management (requires Python 3.8-3.11)
- **`requirements.txt`**: Traditional pip dependencies list (maintained for Docker compatibility)
- Both files should be kept in sync when dependencies change

### Module Organization
- **`solution-guidance/cslib.py`**: Core data ingestion and feature engineering
  - `fetch_data(data_dir)`: Load and normalize JSON invoices into DataFrame
  - `convert_to_ts(df, country=None)`: Aggregate transactions by day for time-series format
  - `fetch_ts(data_dir, clean=False)`: Cache CSV versions of time-series data in `ts-data/` subdirectory
  - `engineer_features(df, training=True)`: Create lagged features (7, 14, 28, 70 days) for revenue prediction

- **`solution-guidance/model.py`**: Model training, loading, and prediction
  - `model_train(data_dir, test=False)`: Train separate models per country using GridSearchCV over Random Forest and SVR
  - `model_load(prefix='sl', data_dir, training=True)`: Load serialized models and feature matrices
  - `model_predict(country, year, month, day, ...)`: Predict 30-day revenue for specified date/country
  - Models are saved as `models/sl-{country}_{version}_{algorithm}.joblib` or `test-*` for test mode

- **`solution-guidance/logger.py`**: CSV-based logging for train/predict events
  - `update_train_log(...)`: Records RMSE, baseline, selected model, runtime to `logs/train.log`
  - `update_predict_log(...)`: Records predictions to `logs/predict.log`
  - Test mode writes to `train-test.log` and `predict-test.log` to keep test artifacts isolated

- **`app.py`**: Flask API with `/health`, `/train`, and `/predict` endpoints
  - Dynamically adds `solution-guidance/` to `sys.path` for module imports
  - Returns JSON responses with proper error handling

- **`monitor.py`**: Post-production analysis utilities
  - `summarize(log_dir)`: Aggregate train/predict event counts and basic stats
  - `detect_anomalies(log_dir)`: Flag out-of-bounds or zero-variance predictions

### Data Flow
1. **Ingestion**: `fetch_data()` reads all `*.json` files from `cs-train/` or `cs-production/`, normalizes column names (handles `StreamID`/`stream_id`, `TimesViewed`/`times_viewed`, `total_price`/`price` variations), removes non-digits from invoice IDs, and adds `invoice_date` column
2. **Time-Series Conversion**: `convert_to_ts()` creates daily aggregates (purchases, unique_invoices, unique_streams, total_views, revenue) using date ranges to fill gaps
3. **Caching**: `fetch_ts()` saves/loads CSVs in `{data_dir}/ts-data/` subdirectory. Uses top 10 countries by revenue plus an "all" aggregate. Delete `ts-data/` or use `clean=True` to regenerate
4. **Feature Engineering**: `engineer_features()` creates lagged revenue features (7, 14, 28, 70 days back) and a 30-day forward revenue target. When `training=False`, returns all engineered data; when `training=True`, trims data without sufficient history
5. **Training**: GridSearchCV selects best model (RF vs SVR) using 3-fold CV, then retrains on full dataset. Logs RMSE vs baseline (mean predictor) to CSV
6. **Prediction**: Looks up engineered features for specified date in cached data, applies loaded model

### Key Design Patterns
- **Per-Country Models**: Separate model trained for each of top 10 countries plus "all" aggregate. This allows country-specific patterns but requires matching country names between training and prediction
- **Test Mode Isolation**: `test=True` flag throughout the codebase subsets data (30%), limits countries to `['all', 'united_kingdom']`, and writes to separate log/model files prefixed with `test-`
- **Lazy Data Loading**: `fetch_ts()` caches CSV files on first run to speed up subsequent model training/loading. Cached files include schema: `date, purchases, unique_invoices, unique_streams, total_views, year_month, revenue`
- **Date-Based Prediction**: Prediction requires exact date match in engineered feature matrix. Models don't interpolate - they look up pre-computed features for the requested date

## Common Workflows

### Adding a New Feature
1. Modify `engineer_features()` in `cslib.py` to create new lagged or rolling window features
2. Delete cached `ts-data/` directories: `cs-train/ts-data/` and `cs-production/ts-data/`
3. Increment `MODEL_VERSION` and update `MODEL_VERSION_NOTE` in `model.py`
4. Retrain: `python -c "import model; model.model_train('cs-train')"`
5. Add corresponding test in `tests/test_model.py` to verify feature shape

### Debugging Prediction Errors
- Check that target date exists in feature matrix: predictions only work for dates with engineered features
- Verify country name matches training data (use lowercase with underscores: `united_kingdom`, not `United Kingdom`)
- Inspect logs: `cat logs/predict.log` or `python -c "import monitor; print(monitor.summarize())"`
- Common error: "date not in range" means requested date is outside training data boundaries or lacks sufficient history for lagged features

### Re-training on New Data
1. Add new JSON files to `cs-train/` following schema: `country, customer_id, day, invoice, month, price, stream_id, times_viewed, year`
2. Delete `cs-train/ts-data/` to force re-processing
3. Run training: `python -c "import model; model.model_train('cs-train')"`
4. Models are versioned via `MODEL_VERSION` in code, not filename timestamps - manually increment when algorithm or feature engineering changes

### Running Tests Properly
- Tests use pytest fixtures to create isolated temp directories
- `monkeypatch.chdir(tmp_path)` ensures test artifacts don't pollute the repository
- Tests override `model.MODEL_DIR` to write to temp locations
- Use `test=True` flag in tests to run fast (subset data, limited countries)
- Tests verify: model serialization, log file creation, API response formats, error handling

## Important Constraints

### Data Schema Assumptions
- JSON files may have inconsistent column names (`StreamID` vs `stream_id`, `TimesViewed` vs `times_viewed`, `total_price` vs `price`) - `fetch_data()` normalizes these
- Invoice IDs contain letters that are stripped: `re.sub(r"\D+", "", str(invoice))`
- Dates are constructed from separate `year, month, day` columns and converted to `datetime64[D]`

### Model Versioning
- `MODEL_VERSION` and `MODEL_VERSION_NOTE` are hardcoded constants in `model.py`, not derived from git tags or timestamps
- When changing feature engineering or model architecture, manually increment `MODEL_VERSION` and update note
- Models are saved with version in filename: `sl-{country}_{version}_{algorithm}.joblib`

### Path Handling
- Use `os.path.join()` for cross-platform compatibility (Windows uses backslashes)
- Relative paths are rooted at repository directory
- Cached data lives in subdirectories: `cs-train/ts-data/`, `models/`, `logs/`
- These cached directories are not git-tracked (create during first run)

### Test vs Production Artifacts
- `test=True` creates: `test-*.joblib` models, `train-test.log`, `predict-test.log`
- Production uses: `sl-*.joblib` models, `train.log`, `predict.log`
- Never mix test and production artifacts in the same directory

## Time-Series Specifics

- The system predicts **30-day forward revenue** for a given date
- Features use **lagged windows**: 7, 14, 28, 70 days of historical revenue
- Training uses `TimeSeriesSplit` implied by the 25% test split with `shuffle=True` (note: this violates typical time-series conventions where shuffle should be False, but is the current implementation)
- Missing days in data are filled with zeros by `convert_to_ts()` using `np.arange(start_month, stop_month, dtype='datetime64[D]')`
- Prediction requires exact date match - models don't extrapolate beyond engineered feature range

## Flask API Endpoints

### `/health` (GET)
Returns `{"status": "ok"}` for uptime checks

### `/train` (POST)
Request: `{"data_dir": "cs-train", "test": false}`
Response: `{"status": "trained", "data_dir": "cs-train", "test": false}`
Side effects: Creates/updates models in `models/` directory, appends to `logs/train.log`

### `/predict` (POST)
Request: `{"country": "all", "year": "2018", "month": "01", "day": "05", "test": false}`
Response: `{"country": "all", "target_date": "2018-01-05", "prediction": 123456.78}`
Requires: Models already trained and available in `models/` directory

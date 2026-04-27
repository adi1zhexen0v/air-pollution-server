import os

MONGODB_URI = os.environ.get("DB_URL")

# Dynamic model name loading from selection pipeline
_SELECTION_DIR = os.path.join(os.path.dirname(__file__), "..", "selection")

_FORECASTING_NAME_FILE = os.path.join(
    _SELECTION_DIR, "outputs", "results", "forecasting_best_model_name.txt"
)

def _read_best_model_name(path, default):
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return default

BEST_FORECASTING_MODEL = _read_best_model_name(_FORECASTING_NAME_FILE, "XGBoost")

# Calibration features (for future use when co-location data is available)
CALIBRATION_FEATURES = [
    "pm25_raw", "humidity", "temperature", "pressure",
    "heat_index", "hour", "month", "day_of_week",
]

# Paths to models and scalers from selection pipeline
SELECTION_MODELS_DIR = os.path.join(_SELECTION_DIR, "outputs", "models")
SELECTION_SCALERS_DIR = os.path.join(_SELECTION_DIR, "outputs", "scalers")

# Thresholds
RETRAIN_MIN_SAMPLES = 50
FORECAST_LOOKBACK = 30
FORECAST_HORIZON = 7
FORECAST_MIN_DAYS = 7

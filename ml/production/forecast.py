"""Generate 7-day PM2.5 forecasts using the best model from selection pipeline.

Supports two model types:
- XGBoost: uses lag/rolling features (flat vector)
- LSTM/BiLSTM/CNN-LSTM: uses raw feature sequences (30×8)
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
import pandas as pd
import torch
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BEST_FORECASTING_MODEL,
    FORECAST_HORIZON,
    FORECAST_LOOKBACK,
    FORECAST_MIN_DAYS,
    MONGODB_URI,
    SELECTION_MODELS_DIR,
    SELECTION_SCALERS_DIR,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "selection"))
from utils import aqi_to_ugm3_pm25
from forecasting_selection import ForecastLSTM, ForecastCNNLSTM  # noqa: F401 (needed for torch.load)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Sequence models use raw features (no lags)
SEQ_MODELS = {"LSTM", "BiLSTM", "CNN-LSTM"}
SEQ_FEATURE_COLS = [
    "pm25", "temperature", "humidity", "pressure",
    "month", "day_of_week", "day",
]


def _is_seq_model():
    return BEST_FORECASTING_MODEL in SEQ_MODELS


def get_db():
    client = MongoClient(MONGODB_URI)
    return client["air-pollution"]


# ── Feature engineering ──────────────────────────────────────────────────────


def build_lag_features(df):
    """Build lag + rolling features for XGBoost."""
    df = df.copy()
    for lag in range(1, FORECAST_LOOKBACK + 1):
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    for window in [7, 14, 30]:
        if window <= FORECAST_LOOKBACK:
            df[f"rolling_mean_{window}"] = df["pm25"].shift(1).rolling(window).mean()
    df["rolling_std_7"] = df["pm25"].shift(1).rolling(min(7, FORECAST_LOOKBACK)).std()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    weather_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
        else:
            df[col] = np.nan
    return df


def build_seq_features(df):
    """Build raw temporal/weather features for sequence models (no lags)."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── Model/scaler loading ────────────────────────────────────────────────────


def load_model():
    """Load the best forecasting model."""
    model_name = BEST_FORECASTING_MODEL
    logger.info("Loading forecasting model: %s", model_name)

    prod_models_dir = os.path.join(os.path.dirname(__file__), "models")

    for models_dir in [prod_models_dir, SELECTION_MODELS_DIR]:
        joblib_path = os.path.join(models_dir, "forecasting_best_model.joblib")
        pt_path = os.path.join(models_dir, "forecasting_best_model_full.pt")

        if not _is_seq_model() and os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
            logger.info("Loaded joblib model from %s", joblib_path)
            return model, "sklearn"

        if _is_seq_model() and os.path.exists(pt_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(pt_path, map_location=device, weights_only=False)
            model.eval()
            logger.info("Loaded PyTorch model from %s", pt_path)
            return model, "torch"

    raise FileNotFoundError(f"No forecasting model found for {model_name}")


def load_scaler():
    """Load the correct scaler based on model type."""
    # Sequence models use forecasting_seq_scaler.joblib
    # XGBoost uses forecasting_scaler.joblib
    scaler_name = "forecasting_seq_scaler.joblib" if _is_seq_model() else "forecasting_scaler.joblib"

    for scalers_dir in [
        os.path.join(os.path.dirname(__file__), "models"),
        SELECTION_SCALERS_DIR,
    ]:
        path = os.path.join(scalers_dir, scaler_name)
        if os.path.exists(path):
            logger.info("Loaded scaler: %s", path)
            return joblib.load(path)

    raise FileNotFoundError(f"Scaler not found: {scaler_name}")


# ── Main forecast ────────────────────────────────────────────────────────────


def _fetch_ref_weather(db, cutoff):
    """Fetch reference station weather data (device-independent, queried once)."""
    ref_pipeline = [
        {"$match": {"createdAt": {"$gte": cutoff}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
            "ref_temperature": {"$avg": "$temperature"},
            "ref_humidity": {"$avg": "$humidity"},
            "ref_pressure": {"$avg": "$pressure"},
        }},
        {"$sort": {"_id": 1}},
    ]
    return {d["_id"]: d for d in db["referencemeasurements"].aggregate(ref_pipeline)}


def _forecast_for_sensor(db, device_id, model, model_type, scaler, cutoff, ref_daily):
    """Generate forecast for a single sensor."""
    measurements_col = db["measurements"]
    forecasts_col = db["forecasts"]

    pipeline = [
        {"$match": {"deviceId": device_id, "createdAt": {"$gte": cutoff}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
            "pm25": {"$avg": {"$ifNull": ["$pm25_calibrated", "$pm25_raw"]}},
            "temperature": {"$avg": "$temperature"},
            "humidity": {"$avg": "$humidity"},
            "pressure": {"$avg": "$pressure"},
        }},
        {"$sort": {"_id": 1}},
    ]

    daily_docs = list(measurements_col.aggregate(pipeline))

    if not daily_docs:
        logger.warning("No data for %s, skipping", device_id)
        return {"station_id": device_id, "status": "no_data"}

    records = []
    for doc in daily_docs:
        date_str = doc["_id"]
        ref = ref_daily.get(date_str, {})
        records.append({
            "date": pd.Timestamp(date_str),
            "pm25": doc["pm25"],
            "temperature": doc.get("temperature") or ref.get("ref_temperature"),
            "humidity": doc.get("humidity") or ref.get("ref_humidity"),
            "pressure": doc.get("pressure") or ref.get("ref_pressure"),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["pm25"])
    available_days = len(df)

    if available_days < FORECAST_MIN_DAYS:
        logger.warning("Insufficient data for %s: %d days (need %d)", device_id, available_days, FORECAST_MIN_DAYS)
        return {"station_id": device_id, "status": "insufficient_data", "days_available": available_days}

    # Padding: if < LOOKBACK days, pad beginning with period mean
    padded = False
    if available_days < FORECAST_LOOKBACK:
        pad_count = FORECAST_LOOKBACK - available_days
        mean_row = df.mean(numeric_only=True)
        pad_rows = []
        first_date = df["date"].iloc[0]
        for i in range(pad_count, 0, -1):
            row = mean_row.to_dict()
            row["date"] = first_date - timedelta(days=i)
            pad_rows.append(row)
        df = pd.concat([pd.DataFrame(pad_rows), df], ignore_index=True)
        padded = True

    pad_info = f", padded" if padded else ""
    logger.info("Generating forecast for %s (%d days available%s)", device_id, available_days, pad_info)

    # ── Predict ──────────────────────────────────────────────────────────
    try:
        if _is_seq_model():
            predictions_raw = _predict_seq(df, model, scaler)
        else:
            predictions_raw = _predict_xgboost(df, model, scaler)
    except Exception as e:
        logger.error("Prediction failed for %s: %s", device_id, e)
        return {"station_id": device_id, "status": "error", "message": str(e)}

    # Build predictions array
    last_date = df["date"].iloc[-1]
    predictions = []
    pred_values = predictions_raw.flatten()[:FORECAST_HORIZON]
    for i, val in enumerate(pred_values):
        predictions.append({
            "day": i + 1,
            "date": last_date + timedelta(days=i + 1),
            "pm25": round(max(0, float(val)), 2),
        })

    # Insert forecast
    forecast_doc = {
        "created_at": datetime.utcnow(),
        "station_id": device_id,
        "model_type": BEST_FORECASTING_MODEL,
        "predictions": predictions,
        "input_days_used": available_days,
        "padded": padded,
    }
    forecasts_col.insert_one(forecast_doc)

    return {
        "station_id": device_id,
        "status": "ok",
        "model": BEST_FORECASTING_MODEL,
        "days_used": available_days,
        "padded": padded,
        "predictions": predictions,
    }


def run_forecast():
    db = get_db()

    # Load model and scaler once
    try:
        model, model_type = load_model()
        scaler = load_scaler()
    except Exception as e:
        logger.error("Failed to load model/scaler: %s", e)
        return {"status": "error", "message": str(e)}

    cutoff = datetime.utcnow() - timedelta(days=60)

    # Get unique sensor deviceIds (exclude reference station)
    device_ids = db["measurements"].distinct("deviceId", {
        "createdAt": {"$gte": cutoff},
        "deviceId": {"$ne": "Reference-Station"},
    })

    if not device_ids:
        logger.warning("No sensors with recent measurements")
        return {"status": "no_data"}

    logger.info("Found %d sensors to forecast: %s", len(device_ids), device_ids)

    # Fetch reference weather once (device-independent)
    ref_daily = _fetch_ref_weather(db, cutoff)

    results = []
    for device_id in sorted(device_ids):
        result = _forecast_for_sensor(db, device_id, model, model_type, scaler, cutoff, ref_daily)
        results.append(result)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info("Forecast complete: %d/%d sensors succeeded", ok_count, len(results))

    return {"status": "ok", "forecasts": results}


def _predict_xgboost(df, model, scaler):
    """Predict using XGBoost with lag/rolling features."""
    df = build_lag_features(df)
    last_row = df.iloc[-1:]
    feature_cols = [c for c in last_row.columns if c not in ["date", "pm25"]]
    X = last_row[feature_cols].fillna(0).values

    if X.shape[1] != scaler.n_features_in_:
        expected = scaler.n_features_in_
        if X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])), constant_values=0)
        else:
            X = X[:, :expected]

    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def _predict_seq(df, model, scaler):
    """Predict using LSTM/BiLSTM/CNN-LSTM with raw feature sequences."""
    df = build_seq_features(df)

    # Use available SEQ_FEATURE_COLS (wind_speed may be missing)
    available = [c for c in SEQ_FEATURE_COLS if c in df.columns]
    if "wind_speed" in df.columns and "wind_speed" not in available:
        available.append("wind_speed")

    # Take last LOOKBACK days
    seq_data = df[available].tail(FORECAST_LOOKBACK).fillna(0).values.astype(np.float64)

    # Align features with scaler
    if seq_data.shape[1] != scaler.n_features_in_:
        expected = scaler.n_features_in_
        if seq_data.shape[1] < expected:
            seq_data = np.pad(seq_data, ((0, 0), (0, expected - seq_data.shape[1])), constant_values=0)
        else:
            seq_data = seq_data[:, :expected]

    # Scale and reshape to (1, LOOKBACK, n_features)
    seq_scaled = scaler.transform(seq_data)
    x_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()

    # Inverse-transform PM2.5 predictions (column 0 in scaler)
    pm25_mean = scaler.mean_[0]
    pm25_scale = scaler.scale_[0]
    return pred_scaled * pm25_scale + pm25_mean


if __name__ == "__main__":
    result = run_forecast()
    print(json.dumps(result, default=str))

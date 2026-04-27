"""Calibrate a single ESP32 measurement using the active MLR model."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
from bson import ObjectId
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(__file__))
from config import MONGODB_URI, SELECTION_MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_db():
    client = MongoClient(MONGODB_URI)
    return client["air-pollution"]


def load_calibration_model(db, sensor_id):
    """Load active calibration model for a sensor.

    Returns (intercept, coef_dict, model_id) or falls back to selection joblib.
    """
    doc = db["calibrationmodels"].find_one(
        {"sensor_id": sensor_id, "is_active": True},
        sort=[("trained_at", -1)],
    )
    if doc:
        coeffs = doc["coefficients"]
        logger.info("Using trained model %s for sensor %s", doc["_id"], sensor_id)
        return coeffs["intercept"], coeffs, doc["_id"]

    # Fallback to selection pipeline model
    try:
        import joblib

        model_path = os.path.join(SELECTION_MODELS_DIR, "calibration_mlr.joblib")
        model = joblib.load(model_path)
        feature_names = [
            "pm25_raw", "pm10_raw", "humidity", "temperature",
            "pressure", "wind_speed", "month", "day_of_week",
        ]
        coef_dict = {"intercept": float(model.intercept_)}
        for name, coef in zip(feature_names, model.coef_):
            coef_dict[name] = float(coef)
        logger.info("Using fallback selection model from %s", model_path)
        return model.intercept_, coef_dict, None
    except Exception as e:
        logger.warning("No model available: %s", e)
        return None, None, None


def predict_mlr(intercept, coef_dict, features):
    """Predict using MLR coefficients stored as a dict."""
    result = intercept
    for name, value in features.items():
        if name in coef_dict and value is not None:
            result += coef_dict[name] * value
    return result


def calibrate_measurement(measurement_id):
    db = get_db()
    measurements = db["measurements"]
    ref_measurements = db["referencemeasurements"]

    # 1. Fetch measurement
    doc = measurements.find_one({"_id": ObjectId(measurement_id)})
    if not doc:
        logger.error("Measurement %s not found", measurement_id)
        return {"status": "error", "message": "Measurement not found"}

    sensor_id = doc.get("sensor_id", doc.get("deviceId", "ESP32-Unit1"))

    # 2. Load model
    intercept, coef_dict, model_id = load_calibration_model(db, sensor_id)
    if intercept is None:
        logger.warning("No calibration model found, skipping")
        return {"status": "skipped", "message": "No calibration model available"}

    # 3. Build feature vector
    now = doc.get("createdAt", datetime.utcnow())
    features = {
        "pm25_raw": doc.get("pm25_raw"),
        "pm10_raw": doc.get("pm10_raw"),
        "humidity": doc.get("humidity"),
        "temperature": doc.get("temperature"),
        "pressure": doc.get("pressure"),
        "wind_speed": None,
        "month": now.month,
        "day_of_week": now.weekday(),
    }

    # Fill missing weather from latest reference measurement
    if features["humidity"] is None or features["temperature"] is None:
        ref = ref_measurements.find_one(sort=[("createdAt", -1)])
        if ref:
            for field in ["humidity", "temperature", "pressure"]:
                if features[field] is None:
                    features[field] = ref.get(field)

    # 4. Predict
    pm25_calibrated = predict_mlr(intercept, coef_dict, features)
    pm25_calibrated = max(0, pm25_calibrated)

    # Estimate pm10 calibrated (simple ratio if pm25_raw > 0)
    pm10_calibrated = None
    if doc.get("pm10_raw") and doc.get("pm25_raw") and doc["pm25_raw"] > 0:
        ratio = pm25_calibrated / doc["pm25_raw"]
        pm10_calibrated = max(0, doc["pm10_raw"] * ratio)

    # 5. Update measurement
    update_fields = {
        "pm25_calibrated": round(pm25_calibrated, 2),
        "calibration_timestamp": datetime.utcnow(),
    }
    if pm10_calibrated is not None:
        update_fields["pm10_calibrated"] = round(pm10_calibrated, 2)
    if model_id:
        update_fields["calibrated_by_model"] = model_id

    measurements.update_one(
        {"_id": ObjectId(measurement_id)},
        {"$set": update_fields},
    )

    result = {
        "status": "ok",
        "measurement_id": measurement_id,
        "pm25_calibrated": update_fields["pm25_calibrated"],
        "pm10_calibrated": update_fields.get("pm10_calibrated"),
    }
    logger.info("Calibrated measurement %s: PM2.5 = %.2f", measurement_id, pm25_calibrated)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement_id", required=True)
    args = parser.parse_args()

    result = calibrate_measurement(args.measurement_id)
    print(json.dumps(result))

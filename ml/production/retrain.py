"""Retrain calibration models for sensors with co-located reference data."""

import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import MONGODB_URI, RETRAIN_MIN_SAMPLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_db():
    client = MongoClient(MONGODB_URI)
    return client["air-pollution"]


def retrain_all():
    db = get_db()
    sensors_col = db["sensors"]
    measurements_col = db["measurements"]
    ref_col = db["referencemeasurements"]
    models_col = db["calibrationmodels"]

    # Find calibration sensors
    cal_sensors = list(sensors_col.find({"is_calibration_sensor": True}))
    if not cal_sensors:
        logger.warning("No calibration sensors found")
        return {"status": "no_sensors", "models_trained": 0}

    results = []

    for sensor in cal_sensors:
        sensor_id = sensor["sensor_id"]
        logger.info("Processing sensor: %s", sensor_id)

        # Fetch sensor measurements
        sensor_docs = list(measurements_col.find(
            {"$or": [{"sensor_id": sensor_id}, {"deviceId": sensor_id}]},
            sort=[("createdAt", 1)],
        ))
        if not sensor_docs:
            logger.warning("No measurements for sensor %s", sensor_id)
            continue

        # Build sensor DataFrame with daily aggregation
        sensor_records = []
        for doc in sensor_docs:
            dt = doc.get("createdAt", datetime.utcnow())
            sensor_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "pm25_raw": doc.get("pm25_raw"),
                "pm10_raw": doc.get("pm10_raw"),
                "humidity": doc.get("humidity"),
                "temperature": doc.get("temperature"),
                "pressure": doc.get("pressure"),
            })

        sensor_df = pd.DataFrame(sensor_records)
        sensor_df = sensor_df.groupby("date").mean().reset_index()

        # Fetch reference measurements (daily aggregation)
        ref_docs = list(ref_col.find(sort=[("createdAt", 1)]))
        if not ref_docs:
            logger.warning("No reference data available")
            continue

        ref_records = []
        for doc in ref_docs:
            dt = doc.get("createdAt", datetime.utcnow())
            ref_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "pm25_ref": doc.get("pm25_raw"),
            })

        ref_df = pd.DataFrame(ref_records)
        ref_df = ref_df.groupby("date").mean().reset_index()

        # Merge on date
        merged = sensor_df.merge(ref_df, on="date", how="inner")
        merged = merged.dropna(subset=["pm25_raw", "pm25_ref"])

        if len(merged) < RETRAIN_MIN_SAMPLES:
            logger.warning(
                "Sensor %s: only %d matched pairs (need %d), skipping",
                sensor_id, len(merged), RETRAIN_MIN_SAMPLES,
            )
            continue

        # Prepare features
        merged["date_dt"] = pd.to_datetime(merged["date"])
        merged["month"] = merged["date_dt"].dt.month
        merged["day_of_week"] = merged["date_dt"].dt.dayofweek

        feature_names = ["pm25_raw", "pm10_raw", "humidity", "temperature",
                         "pressure", "month", "day_of_week"]
        available_features = [f for f in feature_names if f in merged.columns]
        merged_clean = merged.dropna(subset=available_features + ["pm25_ref"])

        X = merged_clean[available_features].values
        y = merged_clean["pm25_ref"].values

        # Chronological 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        logger.info(
            "Sensor %s: R2=%.4f, RMSE=%.2f, MAE=%.2f (%d samples)",
            sensor_id, r2, rmse, mae, len(X),
        )

        # Build coefficients dict
        coefficients = {"intercept": float(model.intercept_)}
        for name, coef in zip(available_features, model.coef_):
            coefficients[name] = float(coef)

        # Store scaler params for reconstruction
        coefficients["scaler_mean"] = scaler.mean_.tolist()
        coefficients["scaler_scale"] = scaler.scale_.tolist()
        coefficients["feature_names"] = available_features

        # Deactivate old models
        models_col.update_many(
            {"sensor_id": sensor_id, "is_active": True},
            {"$set": {"is_active": False}},
        )

        # Compute sensor age
        install_date = sensor.get("install_date")
        sensor_age_days = None
        if install_date:
            sensor_age_days = (datetime.utcnow() - install_date).days

        # Insert new model
        new_model = {
            "sensor_id": sensor_id,
            "trained_at": datetime.utcnow(),
            "model_type": "MLR",
            "coefficients": coefficients,
            "metrics": {"r2": round(r2, 4), "rmse": round(rmse, 2), "mae": round(mae, 2)},
            "sensor_age_days": sensor_age_days,
            "training_samples": len(X),
            "is_active": True,
        }

        # Save joblib backup
        model_dir = os.path.join(os.path.dirname(__file__), "models", sensor_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "calibration_mlr.joblib")
        joblib.dump({"model": model, "scaler": scaler, "features": available_features}, model_path)
        new_model["model_path"] = model_path

        models_col.insert_one(new_model)
        logger.info("Saved new model for sensor %s", sensor_id)

        results.append({
            "sensor_id": sensor_id,
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "samples": len(X),
        })

    summary = {"status": "ok", "models_trained": len(results), "results": results}
    logger.info("Retrain complete: %d models trained", len(results))
    return summary


if __name__ == "__main__":
    result = retrain_all()
    print(json.dumps(result, default=str))

"""Virtual co-location calibration pipeline.

Uses Inverse Distance Weighting (IDW) to interpolate a virtual sensor at the
reference station location from 3 IoT sensors, then trains 9 calibration models
on the paired (virtual_raw, reference_actual) data.

Usage:
    python virtual_colocation.py
"""

import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SENSORS = {
    "Sensor-1": {"lat": 51.0999, "lon": 71.4016},
    "Sensor-2": {"lat": 51.1013, "lon": 71.4296},
    "Sensor-3": {"lat": 51.1138, "lon": 71.4301},
}

REFERENCE_STATION = {"lat": 51.158041944444, "lon": 71.415435}

FEATURE_COLS = [
    "virtual_pm25", "virtual_humidity", "virtual_temperature",
    "virtual_pressure", "virtual_heat_index",
    "hour", "month", "day_of_week",
]
TARGET_COL = "ref_pm25"

MIN_RECORDS_PER_HOUR = 3
IDW_POWER = 2
LSTM_WINDOW = 6
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# ── Utility functions (standalone copies) ────────────────────────────────────


_PM25_BREAKPOINTS = [
    (0, 50, 0.0, 12.0),
    (50, 100, 12.0, 35.4),
    (100, 150, 35.4, 55.4),
    (150, 200, 55.4, 150.4),
    (200, 300, 150.4, 250.4),
    (300, 400, 250.4, 350.4),
    (400, 500, 350.4, 500.4),
]


def aqi_to_ugm3_pm25(aqi):
    """Convert US EPA PM2.5 AQI index to ug/m3 concentration."""
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return None
    aqi = float(aqi)
    if aqi < 0:
        return 0.0
    for aqi_lo, aqi_hi, conc_lo, conc_hi in _PM25_BREAKPOINTS:
        if aqi_lo <= aqi <= aqi_hi:
            return (conc_hi - conc_lo) / (aqi_hi - aqi_lo) * (aqi - aqi_lo) + conc_lo
    aqi_lo, aqi_hi, conc_lo, conc_hi = _PM25_BREAKPOINTS[-1]
    return (conc_hi - conc_lo) / (aqi_hi - aqi_lo) * (aqi - aqi_lo) + conc_lo


def compute_heat_index(temperature_c, humidity_pct):
    """Compute heat index using NOAA formula (Celsius)."""
    if temperature_c is None or humidity_pct is None:
        return None
    if isinstance(temperature_c, float) and math.isnan(temperature_c):
        return None
    if isinstance(humidity_pct, float) and math.isnan(humidity_pct):
        return None
    t = temperature_c * 9.0 / 5.0 + 32.0
    rh = float(humidity_pct)
    hi = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    if hi < 80:
        return (hi - 32.0) * 5.0 / 9.0
    hi = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh
        - 0.22475541 * t * rh - 0.00683783 * t * t
        - 0.05481717 * rh * rh + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh
    )
    if rh < 13 and 80 <= t <= 112:
        hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
    elif rh > 85 and 80 <= t <= 87:
        hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return (hi - 32.0) * 5.0 / 9.0


def compute_metrics(y_true, y_pred):
    """Compute R2, RMSE, and MAE."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ── IDW functions ────────────────────────────────────────────────────────────


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_idw_weights(sensor_coords, target_coord, power=2):
    """Compute IDW weights for each sensor relative to target.

    Returns dict {sensor_name: {"distance_km": float, "weight": float}}.
    """
    distances = {}
    for name, coord in sensor_coords.items():
        d = haversine_km(coord["lat"], coord["lon"],
                         target_coord["lat"], target_coord["lon"])
        distances[name] = d

    inv_d = {name: 1.0 / d ** power for name, d in distances.items()}
    total = sum(inv_d.values())
    weights = {name: v / total for name, v in inv_d.items()}

    result = {}
    for name in sensor_coords:
        result[name] = {
            "distance_km": round(distances[name], 3),
            "weight": round(weights[name], 6),
        }
    return result


# ── Neural network architectures ─────────────────────────────────────────────


class CalibrationANN(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CalibrationLSTM(nn.Module):
    def __init__(self, input_dim=8, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        d = 2 if bidirectional else 1
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True,
                             bidirectional=bidirectional)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64 * d, 32, batch_first=True,
                             bidirectional=bidirectional)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * d, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# ── Database ─────────────────────────────────────────────────────────────────


def get_db():
    """Connect to MongoDB and return the database handle."""
    load_dotenv()
    uri = os.environ.get("MONGODB_URI") or os.environ.get("DB_URL")
    if not uri:
        raise ValueError("Set MONGODB_URI or DB_URL environment variable")
    client = MongoClient(uri)
    db_name = uri.rsplit("/", 1)[-1].split("?")[0] if "/" in uri else "air-pollution"
    db = client[db_name]
    logger.info("Connected to MongoDB database: %s", db_name)
    return client, db


def discover_device_mapping(db):
    """Discover actual deviceIds and map them to known sensor coordinates.

    Returns dict: {actual_deviceId: sensor_name}
    """
    device_ids = db["measurements"].distinct("deviceId")
    logger.info("Found deviceIds in measurements: %s", device_ids)

    mapping = {}
    for did in device_ids:
        sample = db["measurements"].find_one(
            {"deviceId": did, "latitude": {"$ne": None}, "longitude": {"$ne": None}},
            {"latitude": 1, "longitude": 1},
        )
        if sample is None:
            logger.warning("No lat/lon found for deviceId=%s, skipping", did)
            continue

        lat, lon = sample["latitude"], sample["longitude"]
        best_name = None
        best_dist = float("inf")
        for name, coord in SENSORS.items():
            d = haversine_km(lat, lon, coord["lat"], coord["lon"])
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist < 1.0:  # within 1 km
            mapping[did] = best_name
            logger.info("  %s → %s (%.3f km apart)", did, best_name, best_dist)
        else:
            logger.warning("  %s at (%.4f, %.4f) — no match within 1km (nearest=%s, %.1fkm)",
                           did, lat, lon, best_name, best_dist)

    if len(mapping) < 3:
        logger.warning("Only %d of 3 sensors matched! Mapped: %s", len(mapping), mapping)

    return mapping


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_sensor_data(db):
    """Fetch all IoT sensor measurements from MongoDB."""
    cursor = db["measurements"].find(
        {},
        {
            "createdAt": 1, "deviceId": 1,
            "pm1_raw": 1, "pm25_raw": 1, "pm10_raw": 1,
            "temperature": 1, "humidity": 1, "pressure": 1, "heat_index": 1,
            "latitude": 1, "longitude": 1,
        },
    ).sort("createdAt", 1)

    docs = list(cursor)
    df = pd.DataFrame(docs)
    logger.info("Fetched %d sensor records", len(df))

    # Filter bad pressure readings (BMP280 zeros)
    before = len(df)
    df = df[df["pressure"].notna() & (df["pressure"] > 300)]
    logger.info("Removed %d records with invalid pressure (<=300 or null)", before - len(df))

    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    return df


def fetch_reference_data(db):
    """Fetch reference station data from MongoDB."""
    cursor = db["referencemeasurements"].find(
        {"deviceId": "Reference-Station"},
        {
            "createdAt": 1,
            "pm25_raw": 1, "pm25_aqi": 1, "pm25_ugm3": 1,
            "temperature": 1, "humidity": 1, "pressure": 1, "heat_index": 1,
        },
    ).sort("createdAt", 1)

    docs = list(cursor)
    df = pd.DataFrame(docs)
    logger.info("Fetched %d reference records", len(df))

    # Print sample
    logger.info("Sample reference records (first 10):")
    sample_cols = ["createdAt", "pm25_raw", "pm25_aqi", "pm25_ugm3", "temperature", "humidity"]
    available = [c for c in sample_cols if c in df.columns]
    print(df[available].head(10).to_string(index=False))

    # Resolve PM2.5: prefer pm25_ugm3, fallback to AQI conversion
    if "pm25_ugm3" in df.columns and df["pm25_ugm3"].notna().any():
        df["ref_pm25"] = df["pm25_ugm3"].combine_first(
            df["pm25_aqi"].apply(aqi_to_ugm3_pm25) if "pm25_aqi" in df.columns
            else pd.Series([None] * len(df))
        )
        ugm3_count = df["pm25_ugm3"].notna().sum()
        aqi_count = df["ref_pm25"].notna().sum() - ugm3_count
        logger.info("PM2.5 source: %d from pm25_ugm3, %d from AQI conversion", ugm3_count, aqi_count)
    elif "pm25_aqi" in df.columns:
        df["ref_pm25"] = df["pm25_aqi"].apply(aqi_to_ugm3_pm25)
        logger.info("PM2.5 source: all from AQI conversion (pm25_ugm3 not available)")
    else:
        logger.error("No PM2.5 data available in reference measurements!")
        sys.exit(1)

    df = df.dropna(subset=["ref_pm25"])
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    logger.info("Reference records with valid PM2.5: %d", len(df))
    return df


# ── Aggregation & alignment ──────────────────────────────────────────────────


def aggregate_hourly(df, value_cols, min_records=MIN_RECORDS_PER_HOUR):
    """Aggregate 10-min sensor data to hourly means per device."""
    df = df.copy()
    df["hour_floor"] = df["createdAt"].dt.floor("h")

    grouped = df.groupby(["deviceId", "hour_floor"])
    counts = grouped.size().reset_index(name="n_records")
    means = grouped[value_cols].mean().reset_index()

    merged = means.merge(counts, on=["deviceId", "hour_floor"])
    before = len(merged)
    merged = merged[merged["n_records"] >= min_records]
    logger.info("Hourly aggregation: %d → %d (dropped %d with < %d records)",
                before, len(merged), before - len(merged), min_records)
    return merged


def aggregate_reference_hourly(df_ref):
    """Aggregate reference data to hourly means."""
    df_ref = df_ref.copy()
    df_ref["hour_floor"] = df_ref["createdAt"].dt.floor("h")
    value_cols = ["ref_pm25", "temperature", "humidity", "pressure", "heat_index"]
    available = [c for c in value_cols if c in df_ref.columns]
    hourly = df_ref.groupby("hour_floor")[available].mean().reset_index()

    # Rename to avoid collision with sensor columns
    rename_map = {c: f"ref_{c}" if not c.startswith("ref_") and c != "hour_floor" else c
                  for c in hourly.columns}
    hourly = hourly.rename(columns=rename_map)

    logger.info("Reference hourly records: %d", len(hourly))
    return hourly


def align_timestamps(sensor_hourly, ref_hourly, device_mapping):
    """Inner join sensor hourly data with reference, requiring all 3 sensors.

    Returns DataFrame with one row per aligned hour.
    """
    sensor_names = list(set(device_mapping.values()))

    # Pivot sensor data: one column per sensor per variable
    pivoted_frames = []
    for device_id, sensor_name in device_mapping.items():
        dev_df = sensor_hourly[sensor_hourly["deviceId"] == device_id].copy()
        dev_df = dev_df.set_index("hour_floor")
        rename = {}
        for col in dev_df.columns:
            if col not in ("deviceId", "n_records"):
                rename[col] = f"{sensor_name}_{col}"
        dev_df = dev_df.rename(columns=rename)
        dev_df = dev_df.drop(columns=["deviceId", "n_records"], errors="ignore")
        pivoted_frames.append(dev_df)

    # Inner join all sensors
    aligned = pivoted_frames[0]
    for frame in pivoted_frames[1:]:
        aligned = aligned.join(frame, how="inner")

    logger.info("Hours with all %d sensors: %d", len(sensor_names), len(aligned))

    # Join with reference
    ref_indexed = ref_hourly.set_index("hour_floor")
    aligned = aligned.join(ref_indexed, how="inner")

    logger.info("Aligned hours (all sensors + reference): %d", len(aligned))

    if len(aligned) > 0:
        logger.info("Date range: %s → %s",
                     aligned.index.min().strftime("%Y-%m-%d %H:%M"),
                     aligned.index.max().strftime("%Y-%m-%d %H:%M"))

    return aligned


# ── IDW interpolation ────────────────────────────────────────────────────────


def interpolate_virtual_sensor(aligned_df, device_mapping, idw_info):
    """Apply IDW weights to create virtual sensor readings."""
    df = aligned_df.copy()
    weights = {name: info["weight"] for name, info in idw_info.items()}

    fields = ["pm25_raw", "pm1_raw", "pm10_raw", "temperature", "humidity",
              "pressure", "heat_index"]

    for field in fields:
        virtual_col = f"virtual_{field.replace('_raw', '') if field.endswith('_raw') else field}"
        col_sum = None
        for sensor_name, w in weights.items():
            src_col = f"{sensor_name}_{field}"
            if src_col in df.columns:
                contribution = df[src_col] * w
                col_sum = contribution if col_sum is None else col_sum + contribution
        if col_sum is not None:
            df[virtual_col] = col_sum

    logger.info("Virtual sensor columns created: %s",
                [c for c in df.columns if c.startswith("virtual_")])
    return df


# ── Feature engineering ──────────────────────────────────────────────────────


def engineer_features(df):
    """Add time-based features and ensure heat_index."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek

    # Compute heat_index from virtual temperature and humidity if missing
    if "virtual_heat_index" not in df.columns or df["virtual_heat_index"].isna().all():
        df["virtual_heat_index"] = df.apply(
            lambda r: compute_heat_index(
                r.get("virtual_temperature"), r.get("virtual_humidity")
            ), axis=1,
        )

    # Drop rows with NaN in features or target
    required = FEATURE_COLS + [TARGET_COL]
    available = [c for c in required if c in df.columns]
    df = df.dropna(subset=available)
    df = df.sort_index()

    logger.info("Feature-engineered dataset: %d rows, features=%s", len(df), FEATURE_COLS)
    return df


# ── Sklearn model training ──────────────────────────────────────────────────


def train_sklearn_models(X_train, y_train):
    """Train and return dict of sklearn/xgboost models with timings."""
    models = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 1. MLR
    t0 = time.time()
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    models["MLR"] = (mlr, time.time() - t0)

    # 2. KNN
    t0 = time.time()
    knn_params = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    knn_gs = GridSearchCV(KNeighborsRegressor(), knn_params, cv=kf,
                          scoring="r2", n_jobs=-1)
    knn_gs.fit(X_train, y_train)
    models["KNN"] = (knn_gs.best_estimator_, time.time() - t0)
    logger.info("KNN best params: %s", knn_gs.best_params_)

    # 3. SVR
    t0 = time.time()
    svr_params = {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["rbf"],
    }
    svr_gs = GridSearchCV(SVR(), svr_params, cv=kf, scoring="r2", n_jobs=-1)
    svr_gs.fit(X_train, y_train)
    models["SVR"] = (svr_gs.best_estimator_, time.time() - t0)
    logger.info("SVR best params: %s", svr_gs.best_params_)

    # 4. RF
    t0 = time.time()
    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED),
        rf_params, n_iter=20, cv=kf, scoring="r2",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_rs.fit(X_train, y_train)
    models["RF"] = (rf_rs.best_estimator_, time.time() - t0)
    logger.info("RF best params: %s", rf_rs.best_params_)

    # 5. XGBoost
    t0 = time.time()
    xgb_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    xgb_rs = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_SEED, verbosity=0),
        xgb_params, n_iter=20, cv=kf, scoring="r2",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb_rs.fit(X_train, y_train)
    models["XGBoost"] = (xgb_rs.best_estimator_, time.time() - t0)
    logger.info("XGBoost best params: %s", xgb_rs.best_params_)

    # 6. DTR
    t0 = time.time()
    dtr_params = {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    }
    dtr_gs = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_SEED),
        dtr_params, cv=kf, scoring="r2", n_jobs=-1,
    )
    dtr_gs.fit(X_train, y_train)
    models["DTR"] = (dtr_gs.best_estimator_, time.time() - t0)
    logger.info("DTR best params: %s", dtr_gs.best_params_)

    return models


# ── PyTorch training helpers ─────────────────────────────────────────────────


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_ann(X_train, y_train, X_val, y_val):
    """Train the ANN model with early stopping."""
    device = _get_device()
    model = CalibrationANN(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("ANN early stopping at epoch %d", epoch + 1)
                break

    elapsed = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


def _create_sliding_windows(X, y, window_size):
    """Create sliding windows for LSTM/BiLSTM."""
    Xw, yw = [], []
    for i in range(len(X) - window_size + 1):
        Xw.append(X[i:i + window_size])
        yw.append(y[i + window_size - 1])
    if not Xw:
        return np.array([]).reshape(0, window_size, X.shape[1]), np.array([])
    return np.array(Xw), np.array(yw)


def train_lstm_model(X_train_w, y_train_w, X_val_w, y_val_w,
                     input_dim, bidirectional=False):
    """Train LSTM or BiLSTM with early stopping."""
    device = _get_device()
    model = CalibrationLSTM(input_dim=input_dim,
                            bidirectional=bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_w, dtype=torch.float32),
        torch.tensor(y_train_w, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    X_val_t = torch.tensor(X_val_w, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_w, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(X_val_t) > 0:
                val_loss = criterion(model(X_val_t), y_val_t).item()
            else:
                val_loss = float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                name = "BiLSTM" if bidirectional else "LSTM"
                logger.info("%s early stopping at epoch %d", name, epoch + 1)
                break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


# ── Cross-validation ─────────────────────────────────────────────────────────


def cross_validate_sklearn(X, y, models_dict):
    """Run KFold CV for sklearn models."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {}

    for name, (model, _) in models_dict.items():
        r2s, rmses, maes = [], [], []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            clone = type(model)(**model.get_params())
            clone.fit(X_tr, y_tr)
            y_pred = clone.predict(X_va)
            m = compute_metrics(y_va, y_pred)
            r2s.append(m["r2"])
            rmses.append(m["rmse"])
            maes.append(m["mae"])

        cv_results[name] = {
            "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
            "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        }
        logger.info("CV %s: R2=%.4f±%.4f  RMSE=%.2f  MAE=%.2f",
                     name, np.mean(r2s), np.std(r2s),
                     np.mean(rmses), np.mean(maes))

    return cv_results


def cross_validate_nn(X_scaled, y, model_class, nn_kwargs, window=None):
    """Run CV for a neural network model.

    KFold for ANN, TimeSeriesSplit for LSTM/BiLSTM.
    """
    if window is not None:
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    r2s, rmses, maes = [], [], []
    device = _get_device()

    for train_idx, val_idx in cv.split(X_scaled):
        X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if window is not None:
            X_tr_w, y_tr_w = _create_sliding_windows(X_tr, y_tr, window)
            X_va_w, y_va_w = _create_sliding_windows(X_va, y_va, window)
            if len(X_tr_w) == 0 or len(X_va_w) == 0:
                continue
            model = model_class(**nn_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            train_ds = TensorDataset(
                torch.tensor(X_tr_w, dtype=torch.float32),
                torch.tensor(y_tr_w, dtype=torch.float32),
            )
            loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
            best_val = float("inf")
            patience_cnt = 0
            best_st = None
            X_va_t = torch.tensor(X_va_w, dtype=torch.float32).to(device)
            y_va_t = torch.tensor(y_va_w, dtype=torch.float32).to(device)
            for _ in range(MAX_EPOCHS):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    criterion(model(xb), yb).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    vl = criterion(model(X_va_t), y_va_t).item()
                if vl < best_val:
                    best_val = vl
                    patience_cnt = 0
                    best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        break
            if best_st:
                model.load_state_dict(best_st)
            model.eval()
            with torch.no_grad():
                preds = model(X_va_t).cpu().numpy()
            m = compute_metrics(y_va_w, preds)
        else:
            # ANN path
            model, _ = train_ann(X_tr, y_tr, X_va, y_va)
            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.tensor(X_va, dtype=torch.float32).to(device)
                ).cpu().numpy()
            m = compute_metrics(y_va, preds)

        r2s.append(m["r2"])
        rmses.append(m["rmse"])
        maes.append(m["mae"])

    return {
        "r2_mean": np.mean(r2s) if r2s else float("nan"),
        "r2_std": np.std(r2s) if r2s else float("nan"),
        "rmse_mean": np.mean(rmses) if rmses else float("nan"),
        "rmse_std": np.std(rmses) if rmses else float("nan"),
        "mae_mean": np.mean(maes) if maes else float("nan"),
        "mae_std": np.std(maes) if maes else float("nan"),
    }


# ── Visualization ────────────────────────────────────────────────────────────


def plot_sensor_map(idw_info, output_dir):
    """01: Map showing sensors, reference, and virtual point with distances."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, coord in SENSORS.items():
        info = idw_info[name]
        ax.scatter(coord["lon"], coord["lat"], s=120, c="dodgerblue",
                   edgecolors="black", zorder=5)
        ax.annotate(f"{name}\nw={info['weight']:.3f}",
                    (coord["lon"], coord["lat"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)

    ref = REFERENCE_STATION
    ax.scatter(ref["lon"], ref["lat"], s=200, c="red", marker="*",
               edgecolors="black", zorder=6)
    ax.annotate("Reference Station\n(AQICN @531799)",
                (ref["lon"], ref["lat"]),
                textcoords="offset points", xytext=(10, -20), fontsize=9,
                color="red")

    ax.scatter(ref["lon"], ref["lat"], s=150, c="orange", marker="D",
               edgecolors="black", zorder=5, alpha=0.7)
    ax.annotate("Virtual Sensor (Sensor-4)",
                (ref["lon"], ref["lat"]),
                textcoords="offset points", xytext=(10, 10), fontsize=9,
                color="orange")

    for name, coord in SENSORS.items():
        info = idw_info[name]
        ax.plot([coord["lon"], ref["lon"]], [coord["lat"], ref["lat"]],
                "k--", alpha=0.3, linewidth=0.8)
        mid_lon = (coord["lon"] + ref["lon"]) / 2
        mid_lat = (coord["lat"] + ref["lat"]) / 2
        ax.text(mid_lon, mid_lat, f"{info['distance_km']:.1f} km",
                fontsize=8, ha="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sensor Network Layout — Virtual Co-location via IDW")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_sensor_map.png"), dpi=150)
    plt.close()


def plot_idw_timeseries(df, output_dir):
    """02: Virtual PM2.5 vs reference PM2.5 time series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full period
    axes[0].plot(df.index, df["virtual_pm25"], label="Virtual PM2.5 (IDW)",
                 color="dodgerblue", linewidth=0.8, alpha=0.8)
    axes[0].plot(df.index, df[TARGET_COL], label="Reference PM2.5",
                 color="red", linestyle="--", linewidth=0.8, alpha=0.8)
    axes[0].set_title("Full Period — Virtual vs Reference PM2.5")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 7-day zoom
    end = df.index.max()
    start = end - pd.Timedelta(days=7)
    zoom = df.loc[start:end]
    axes[1].plot(zoom.index, zoom["virtual_pm25"], label="Virtual PM2.5 (IDW)",
                 color="dodgerblue", linewidth=1.0)
    axes[1].plot(zoom.index, zoom[TARGET_COL], label="Reference PM2.5",
                 color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("7-Day Zoom — Virtual vs Reference PM2.5")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_idw_timeseries.png"), dpi=150)
    plt.close()


def plot_scatter_virtual_vs_ref(df, output_dir):
    """03: Scatter plot of virtual vs reference PM2.5."""
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = df["virtual_pm25"].values, df[TARGET_COL].values

    ax.scatter(x, y, alpha=0.4, s=20, color="steelblue")

    # 1:1 line
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=1.5,
            label=f"Regression (y={z[0]:.2f}x+{z[1]:.2f})")

    r2 = r2_score(y, x)
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Virtual PM2.5 (µg/m³)")
    ax.set_ylabel("Reference PM2.5 (µg/m³)")
    ax.set_title("Virtual Sensor vs Reference — PM2.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_scatter_virtual_vs_ref.png"), dpi=150)
    plt.close()


def plot_bland_altman(df, output_dir):
    """04: Bland-Altman plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    virtual = df["virtual_pm25"].values
    ref = df[TARGET_COL].values
    mean_vals = (virtual + ref) / 2
    diff_vals = virtual - ref
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=20, color="steelblue")
    ax.axhline(mean_diff, color="red", linestyle="-", label=f"Mean diff = {mean_diff:.2f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color="gray", linestyle="--",
               label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.2f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle="--",
               label=f"-1.96 SD = {mean_diff - 1.96 * std_diff:.2f}")

    ax.set_xlabel("Mean of Virtual and Reference (µg/m³)")
    ax.set_ylabel("Difference (Virtual - Reference) (µg/m³)")
    ax.set_title("Bland-Altman Plot — Virtual vs Reference PM2.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_bland_altman.png"), dpi=150)
    plt.close()


def plot_model_comparison(results_df, output_dir):
    """05: Grouped bar chart of CV R² and Final R²."""
    fig, ax = plt.subplots(figsize=(12, 6))
    names = results_df["model"].tolist()
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, results_df["r2_mean"], width, label="CV R² (mean)",
           color="steelblue", yerr=results_df["r2_std"], capsize=3)
    ax.bar(x + width / 2, results_df["r2_final"], width, label="Final R² (test)",
           color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison — CV R² vs Final R²")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_model_comparison.png"), dpi=150)
    plt.close()


def plot_best_model_predictions(timestamps, y_test, y_pred, best_name, output_dir):
    """06: Time series of actual vs predicted for best model on test set."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, y_test, label="Actual (Reference)", color="red",
            linewidth=0.8, alpha=0.8)
    ax.plot(timestamps, y_pred, label=f"Predicted ({best_name})", color="dodgerblue",
            linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Best Model ({best_name}) — Predictions vs Actual on Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_best_model_predictions.png"), dpi=150)
    plt.close()


def plot_residuals(y_pred, residuals, best_name, output_dir):
    """07: Residual plot for best model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted PM2.5 (µg/m³)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Residual Plot — {best_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_residuals.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    """08: Correlation matrix heatmap."""
    cols = FEATURE_COLS + [TARGET_COL]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_correlation_heatmap.png"), dpi=150)
    plt.close()


# ── Export ───────────────────────────────────────────────────────────────────


def export_virtual_data(df, idw_info, device_mapping, output_dir):
    """Export virtual_colocation_data.json matching ReferenceMeasurement schema."""
    records = []
    for ts, row in df.iterrows():
        records.append({
            "pm1_raw": round(float(row.get("virtual_pm1", 0)), 2) if pd.notna(row.get("virtual_pm1")) else None,
            "pm25_raw": round(float(row["virtual_pm25"]), 2),
            "pm10_raw": round(float(row.get("virtual_pm10", 0)), 2) if pd.notna(row.get("virtual_pm10")) else None,
            "pm1_calibrated": None,
            "pm25_calibrated": None,
            "pm10_calibrated": None,
            "pm25_aqi": None,
            "pm25_ugm3": None,
            "temperature": round(float(row["virtual_temperature"]), 2),
            "pressure": round(float(row["virtual_pressure"]), 2),
            "humidity": round(float(row["virtual_humidity"]), 2),
            "heat_index": round(float(row["virtual_heat_index"]), 2) if pd.notna(row.get("virtual_heat_index")) else None,
            "latitude": REFERENCE_STATION["lat"],
            "longitude": REFERENCE_STATION["lon"],
            "satellites": None,
            "deviceId": "Sensor-4",
            "createdAt": ts.isoformat(),
        })

    # Metadata
    distances = {name: info["distance_km"] for name, info in idw_info.items()}
    weights = {name: info["weight"] for name, info in idw_info.items()}

    output = {
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "method": "IDW",
            "idw_power": IDW_POWER,
            "n_records": len(records),
            "observation_period": {
                "start": df.index.min().strftime("%Y-%m-%d"),
                "end": df.index.max().strftime("%Y-%m-%d"),
            },
            "sensors_used": list(device_mapping.keys()),
            "sensor_mapping": {did: sname for did, sname in device_mapping.items()},
            "reference_station": "@531799",
            "distances_km": distances,
            "weights": weights,
        },
        "data": records,
    }

    path = os.path.join(output_dir, "virtual_colocation_data.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Exported %d virtual sensor records to %s", len(records), path)


def export_calibration_results(results_df, best_name, dataset_info, output_dir):
    """Export calibration_results.json."""
    models_list = []
    for _, row in results_df.iterrows():
        models_list.append({
            "name": row["model"],
            "cv_r2_mean": round(float(row["r2_mean"]), 4) if pd.notna(row["r2_mean"]) else None,
            "cv_r2_std": round(float(row["r2_std"]), 4) if pd.notna(row["r2_std"]) else None,
            "cv_rmse_mean": round(float(row["rmse_mean"]), 4) if pd.notna(row["rmse_mean"]) else None,
            "cv_mae_mean": round(float(row["mae_mean"]), 4) if pd.notna(row["mae_mean"]) else None,
            "final_r2": round(float(row["r2_final"]), 4),
            "final_rmse": round(float(row["rmse_final"]), 4),
            "final_mae": round(float(row["mae_final"]), 4),
            "training_time_sec": float(row["training_time_sec"]),
        })

    output = {
        "run_date": datetime.utcnow().isoformat(),
        "method": "virtual_colocation_IDW",
        "dataset_size": dataset_info["total"],
        "models": models_list,
        "best_model": best_name,
        "dataset_info": dataset_info,
    }

    path = os.path.join(output_dir, "calibration_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Exported calibration results to %s", path)


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Connect to MongoDB ────────────────────────────────────────────
    client, db = get_db()

    try:
        # ── 2. Discover device IDs ───────────────────────────────────────
        device_mapping = discover_device_mapping(db)
        if len(device_mapping) < 3:
            logger.error("Need 3 sensors, found %d. Aborting.", len(device_mapping))
            sys.exit(1)

        # Build sensor_coords from actual mapping
        sensor_coords = {}
        for did, sname in device_mapping.items():
            sensor_coords[sname] = SENSORS[sname]

        # ── 3. Compute IDW weights ───────────────────────────────────────
        idw_info = compute_idw_weights(sensor_coords, REFERENCE_STATION, power=IDW_POWER)
        logger.info("IDW weights (power=%d):", IDW_POWER)
        for name, info in idw_info.items():
            logger.info("  %s → Reference: %.3f km, weight: %.6f",
                        name, info["distance_km"], info["weight"])

        # ── 4. Fetch data ────────────────────────────────────────────────
        sensor_df = fetch_sensor_data(db)
        ref_df = fetch_reference_data(db)

        # ── 5. Hourly aggregation ────────────────────────────────────────
        sensor_value_cols = ["pm25_raw", "pm1_raw", "pm10_raw",
                             "temperature", "humidity", "pressure", "heat_index"]
        sensor_hourly = aggregate_hourly(sensor_df, sensor_value_cols)

        # Print per-sensor stats
        for did, sname in device_mapping.items():
            n = len(sensor_hourly[sensor_hourly["deviceId"] == did])
            logger.info("  %s (%s): %d hourly records", sname, did, n)

        ref_hourly = aggregate_reference_hourly(ref_df)

        # ── 6. Temporal alignment ────────────────────────────────────────
        aligned = align_timestamps(sensor_hourly, ref_hourly, device_mapping)
        if len(aligned) < 50:
            logger.error("Insufficient aligned data: %d hours (need >=50). Aborting.",
                         len(aligned))
            sys.exit(1)

        # ── 7. IDW interpolation ─────────────────────────────────────────
        aligned = interpolate_virtual_sensor(aligned, device_mapping, idw_info)

        # ── 8. Feature engineering ───────────────────────────────────────
        df = engineer_features(aligned)
        logger.info("Final dataset: %d rows", len(df))

        # ── 9. Plot IDW-related visualizations ───────────────────────────
        plot_sensor_map(idw_info, OUTPUT_DIR)
        plot_idw_timeseries(df, OUTPUT_DIR)
        plot_scatter_virtual_vs_ref(df, OUTPUT_DIR)
        plot_bland_altman(df, OUTPUT_DIR)
        plot_correlation_heatmap(df, OUTPUT_DIR)
        logger.info("Saved plots 01-04 and 08")

    finally:
        client.close()

    # ── 10. Train/test split & scaling ───────────────────────────────────
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)

    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ── 11. Train sklearn models ─────────────────────────────────────────
    logger.info("Training sklearn/XGBoost models...")
    sklearn_models = train_sklearn_models(X_train, y_train)

    # ── 12. Cross-validate sklearn models ────────────────────────────────
    logger.info("Cross-validating sklearn models...")
    cv_results = cross_validate_sklearn(X_train, y_train, sklearn_models)

    # ── 13. Train ANN ────────────────────────────────────────────────────
    logger.info("Training ANN...")
    ann_model, ann_time = train_ann(X_train, y_train, X_test, y_test)
    logger.info("ANN trained in %.1fs", ann_time)

    logger.info("Cross-validating ANN...")
    cv_ann = cross_validate_nn(X_train, y_train, CalibrationANN,
                               {"input_dim": X_train.shape[1]})
    cv_results["ANN"] = cv_ann

    # ── 14. Train LSTM / BiLSTM ──────────────────────────────────────────
    logger.info("Preparing LSTM sliding windows...")
    X_train_w, y_train_w = _create_sliding_windows(X_train, y_train, LSTM_WINDOW)
    X_test_w, y_test_w = _create_sliding_windows(X_test, y_test, LSTM_WINDOW)

    lstm_model = bilstm_model = None
    lstm_time = bilstm_time = 0.0

    if len(X_train_w) > 0:
        logger.info("Training LSTM (windows: train=%d, test=%d)...",
                     len(X_train_w), len(X_test_w))
        lstm_model, lstm_time = train_lstm_model(
            X_train_w, y_train_w, X_test_w, y_test_w,
            input_dim=X_train.shape[1], bidirectional=False,
        )
        logger.info("Cross-validating LSTM...")
        cv_lstm = cross_validate_nn(
            X_train, y_train, CalibrationLSTM,
            {"input_dim": X_train.shape[1], "bidirectional": False},
            window=LSTM_WINDOW,
        )
        cv_results["LSTM"] = cv_lstm

        logger.info("Training BiLSTM...")
        bilstm_model, bilstm_time = train_lstm_model(
            X_train_w, y_train_w, X_test_w, y_test_w,
            input_dim=X_train.shape[1], bidirectional=True,
        )
        logger.info("Cross-validating BiLSTM...")
        cv_bilstm = cross_validate_nn(
            X_train, y_train, CalibrationLSTM,
            {"input_dim": X_train.shape[1], "bidirectional": True},
            window=LSTM_WINDOW,
        )
        cv_results["BiLSTM"] = cv_bilstm
    else:
        logger.warning("Insufficient data for LSTM windows (need > %d rows)", LSTM_WINDOW)

    # ── 15. Final evaluation on test set ─────────────────────────────────
    results_rows = []
    all_models = {}
    all_predictions = {}
    device = _get_device()

    # Sklearn models
    for name, (model, train_time) in sklearn_models.items():
        y_pred = model.predict(X_test)
        final = compute_metrics(y_test, y_pred)
        cv = cv_results.get(name, {})
        results_rows.append({
            "model": name,
            "r2_mean": cv.get("r2_mean"), "r2_std": cv.get("r2_std"),
            "rmse_mean": cv.get("rmse_mean"), "rmse_std": cv.get("rmse_std"),
            "mae_mean": cv.get("mae_mean"), "mae_std": cv.get("mae_std"),
            "r2_final": final["r2"],
            "rmse_final": final["rmse"],
            "mae_final": final["mae"],
            "training_time_sec": round(train_time, 2),
        })
        all_models[name] = ("sklearn", model)
        all_predictions[name] = y_pred

    # ANN
    ann_model.eval()
    with torch.no_grad():
        ann_pred = ann_model(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        ).cpu().numpy()
    ann_final = compute_metrics(y_test, ann_pred)
    cv = cv_results.get("ANN", {})
    results_rows.append({
        "model": "ANN",
        "r2_mean": cv.get("r2_mean"), "r2_std": cv.get("r2_std"),
        "rmse_mean": cv.get("rmse_mean"), "rmse_std": cv.get("rmse_std"),
        "mae_mean": cv.get("mae_mean"), "mae_std": cv.get("mae_std"),
        "r2_final": ann_final["r2"],
        "rmse_final": ann_final["rmse"],
        "mae_final": ann_final["mae"],
        "training_time_sec": round(ann_time, 2),
    })
    all_models["ANN"] = ("torch", ann_model)
    all_predictions["ANN"] = ann_pred

    # LSTM / BiLSTM
    for name, model, train_time in [("LSTM", lstm_model, lstm_time),
                                     ("BiLSTM", bilstm_model, bilstm_time)]:
        if model is None:
            continue
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.tensor(X_test_w, dtype=torch.float32).to(device)
            ).cpu().numpy()
        final = compute_metrics(y_test_w, pred)
        cv = cv_results.get(name, {})
        results_rows.append({
            "model": name,
            "r2_mean": cv.get("r2_mean"), "r2_std": cv.get("r2_std"),
            "rmse_mean": cv.get("rmse_mean"), "rmse_std": cv.get("rmse_std"),
            "mae_mean": cv.get("mae_mean"), "mae_std": cv.get("mae_std"),
            "r2_final": final["r2"],
            "rmse_final": final["rmse"],
            "mae_final": final["mae"],
            "training_time_sec": round(train_time, 2),
        })
        all_models[name] = ("torch", model)
        all_predictions[name] = pred

    # ── 16. Results table ────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    logger.info("\n" + "=" * 100)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 100)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 100)

    # Determine best model
    valid = results_df.dropna(subset=["r2_mean"])
    if len(valid) > 0:
        best_idx = valid.sort_values(["r2_mean", "rmse_mean"],
                                     ascending=[False, True]).index[0]
        best_row = results_df.loc[best_idx]
    else:
        best_row = results_df.loc[results_df["r2_final"].idxmax()]
    best_name = best_row["model"]
    logger.info("Best model: %s (CV R2=%.4f±%.4f, Final R2=%.4f, RMSE=%.2f)",
                best_name, best_row["r2_mean"], best_row["r2_std"],
                best_row["r2_final"], best_row["rmse_final"])

    # ── 17. Plot model results ───────────────────────────────────────────
    plot_model_comparison(results_df, OUTPUT_DIR)

    # Best model predictions plot
    test_timestamps = df.index[split_idx:]
    if best_name in ("LSTM", "BiLSTM"):
        # LSTM predictions are shorter due to windowing
        pred_ts = test_timestamps[LSTM_WINDOW - 1:]
        best_pred = all_predictions[best_name]
        best_y_test = y_test_w
    else:
        pred_ts = test_timestamps
        best_pred = all_predictions[best_name]
        best_y_test = y_test

    plot_best_model_predictions(pred_ts, best_y_test, best_pred,
                                best_name, OUTPUT_DIR)
    plot_residuals(best_pred, best_y_test - best_pred, best_name, OUTPUT_DIR)
    logger.info("Saved plots 05-08")

    # ── 18. Export results ───────────────────────────────────────────────
    export_virtual_data(df, idw_info, device_mapping, OUTPUT_DIR)

    dataset_info = {
        "total": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": FEATURE_COLS,
    }
    export_calibration_results(results_df, best_name, dataset_info, OUTPUT_DIR)

    # Save best model and scaler
    model_type, best_model_obj = all_models[best_name]
    if model_type == "sklearn":
        joblib.dump(best_model_obj,
                    os.path.join(OUTPUT_DIR, "best_calibration_model.pkl"))
    else:
        torch.save(best_model_obj.state_dict(),
                   os.path.join(OUTPUT_DIR, "best_calibration_model.pkl"))

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    logger.info("Saved best model and scaler to %s", OUTPUT_DIR)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()

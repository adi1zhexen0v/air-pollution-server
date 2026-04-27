"""Calibration model selection pipeline.

Compares 9 models (MLR, KNN, SVR, RF, XGBoost, DTR, ANN, LSTM, BiLSTM)
for calibrating low-cost PM2.5 sensor readings against reference data.

Usage:
    python calibration_selection.py                   # auto-detect CSV or MongoDB
    python calibration_selection.py --csv data/paired.csv
    python calibration_selection.py --mongo            # force MongoDB mode
"""

import argparse
import glob as globmod
import json
import logging
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent for utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    RANDOM_SEED,
    aqi_to_ugm3_pm25,
    compute_heat_index,
    compute_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SCALERS_DIR = os.path.join(OUTPUT_DIR, "scalers")

FEATURE_COLS = [
    "pm25_raw", "humidity", "temperature", "pressure",
    "heat_index", "hour", "month", "day_of_week",
]
TARGET_COL = "pm25_ref"

LSTM_WINDOW = 6
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10


# ── Neural network architectures ──────────────────────────────────────────────


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
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=bidirectional)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64 * d, 32, batch_first=True, bidirectional=bidirectional)
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


# ── Synthetic data generation ─────────────────────────────────────────────────


def generate_synthetic_sensor_data(ref_pm25, humidity, temperature, hour,
                                   sensor_id="Synthetic-1", random_state=42):
    """Generate realistic synthetic PMS5003 readings from reference data.

    Simulates known PMS5003 biases documented in the literature:
    - Baseline overestimation ~40% (Barkjohn et al., 2021)
    - Humidity-dependent bias above 60% RH (Giordano et al., 2021)
    - Random noise with CV ~15% (Sayahi et al., 2019)
    - Diurnal pattern variation (heating peaks)
    """
    rng = np.random.RandomState(random_state)

    ref_pm25 = np.asarray(ref_pm25, dtype=np.float64)
    humidity = np.asarray(humidity, dtype=np.float64)
    hour = np.asarray(hour, dtype=np.int32)

    # 1. Baseline overestimation factor (sensor-specific, ~1.3-1.5)
    sensor_bias = 1.4 + rng.normal(0, 0.05)

    # 2. Humidity-dependent bias (nonlinear, stronger above 60% RH)
    rh_excess = np.maximum(humidity - 60, 0.0)
    humidity_factor = np.where(
        humidity > 60,
        1.0 + 0.005 * rh_excess ** 1.5,
        1.0,
    )

    # 3. Base synthetic PM2.5
    synthetic_pm25 = ref_pm25 * sensor_bias * humidity_factor

    # 4. Diurnal pattern variation (morning 7-9, evening 18-21 heating peaks)
    diurnal_offset = np.where(
        (hour >= 7) & (hour <= 9), 2.0,
        np.where((hour >= 18) & (hour <= 21), 3.0, 0.0),
    )
    synthetic_pm25 = synthetic_pm25 + diurnal_offset

    # 5. Random noise (CV ~15%)
    noise = rng.normal(0, 0.15 * np.abs(synthetic_pm25))
    synthetic_pm25 = synthetic_pm25 + noise

    # 6. Ensure non-negative
    synthetic_pm25 = np.maximum(synthetic_pm25, 0.0)

    return synthetic_pm25


def load_reference_from_mongodb():
    """Load reference station data with weather fields from MongoDB (hourly).

    Returns a DataFrame with columns: datetime, pm25_ref, humidity,
    temperature, pressure.
    """
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    uri = os.environ.get("MONGO_URI") or os.environ.get("DB_URL")
    if not uri:
        raise ValueError("Set MONGO_URI or DB_URL environment variable")

    client = MongoClient(uri)
    db = client["air-pollution"]
    logger.info("Connected to MongoDB")

    # Hourly reference means with weather
    pipeline = [
        {
            "$group": {
                "_id": {
                    "date": {"$dateTrunc": {"date": "$createdAt", "unit": "hour"}},
                },
                "pm25_raw": {"$avg": "$pm25_raw"},
                "pm25_ugm3": {"$avg": "$pm25_ugm3"},
                "temperature": {"$avg": "$temperature"},
                "humidity": {"$avg": "$humidity"},
                "pressure": {"$avg": "$pressure"},
            }
        },
        {"$sort": {"_id.date": 1}},
    ]
    docs = list(db["referencemeasurements"].aggregate(pipeline))
    client.close()

    df = pd.DataFrame([
        {
            "datetime": d["_id"]["date"],
            "pm25_ref_raw": d["pm25_raw"],
            "pm25_ref_ugm3": d.get("pm25_ugm3"),
            "temperature": d.get("temperature"),
            "humidity": d.get("humidity"),
            "pressure": d.get("pressure"),
        }
        for d in docs
    ])

    # Use pm25_ugm3 if available, else convert AQI
    if "pm25_ref_ugm3" in df.columns and df["pm25_ref_ugm3"].notna().any():
        df["pm25_ref"] = df["pm25_ref_ugm3"].fillna(
            df["pm25_ref_raw"].apply(aqi_to_ugm3_pm25)
        )
    else:
        df["pm25_ref"] = df["pm25_ref_raw"].apply(aqi_to_ugm3_pm25)

    # Drop rows missing reference PM2.5 or weather
    df = df.dropna(subset=["pm25_ref", "temperature", "humidity", "pressure"])

    df["date"] = df["datetime"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["datetime"].dt.hour

    logger.info("Loaded %d hourly reference records from MongoDB", len(df))
    return df


def build_synthetic_dataset(ref_df, sensor_id="Synthetic-1", random_state=42):
    """Build a paired calibration dataset from reference data + synthetic sensor.

    Parameters
    ----------
    ref_df : DataFrame
        Must contain: pm25_ref, humidity, temperature, pressure, hour, date.

    Returns
    -------
    DataFrame with columns matching FEATURE_COLS + TARGET_COL.
    """
    synthetic_pm25 = generate_synthetic_sensor_data(
        ref_pm25=ref_df["pm25_ref"].values,
        humidity=ref_df["humidity"].values,
        temperature=ref_df["temperature"].values,
        hour=ref_df["hour"].values,
        sensor_id=sensor_id,
        random_state=random_state,
    )

    df = ref_df[["date", "datetime", "humidity", "temperature", "pressure",
                  "hour", "pm25_ref"]].copy()
    df["pm25_raw"] = synthetic_pm25

    # Sanity check
    ref_mean = ref_df["pm25_ref"].mean()
    ref_std = ref_df["pm25_ref"].std()
    syn_mean = np.mean(synthetic_pm25)
    syn_std = np.std(synthetic_pm25)
    logger.info(
        "Synthetic PM2.5: mean=%.1f, std=%.1f (reference: mean=%.1f, std=%.1f)",
        syn_mean, syn_std, ref_mean, ref_std,
    )
    logger.info("Overestimation ratio: %.2f", syn_mean / ref_mean if ref_mean > 0 else float("nan"))

    return df


# ── Data loading ──────────────────────────────────────────────────────────────


def load_from_csv(csv_path):
    """Load pre-prepared paired CSV.

    Expected columns: date, hour (optional), pm25_raw, humidity, temperature,
    pressure, pm25_ref (reference in ug/m3 or AQI — auto-detect).
    """
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # If pm25_ref values look like AQI (integer-ish, max ~500), convert
    if "pm25_ref" in df.columns and df["pm25_ref"].max() > 300:
        logger.info("Detected AQI values in pm25_ref — converting to ug/m3")
        df["pm25_ref"] = df["pm25_ref"].apply(aqi_to_ugm3_pm25)

    return df


def load_from_mongodb():
    """Load paired sensor + reference data from MongoDB, aggregated to hourly."""
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    uri = os.environ.get("MONGO_URI") or os.environ.get("DB_URL")
    if not uri:
        raise ValueError("Set MONGO_URI or DB_URL environment variable")

    client = MongoClient(uri)
    db = client["air-pollution"]
    logger.info("Connected to MongoDB")

    # Hourly sensor means
    sensor_pipeline = [
        {
            "$group": {
                "_id": {
                    "date": {"$dateTrunc": {"date": "$createdAt", "unit": "hour"}},
                },
                "pm25_raw": {"$avg": "$pm25_raw"},
                "humidity": {"$avg": "$humidity"},
                "temperature": {"$avg": "$temperature"},
                "pressure": {"$avg": "$pressure"},
                "heat_index": {"$avg": "$heat_index"},
            }
        },
        {"$sort": {"_id.date": 1}},
    ]
    sensor_docs = list(db["measurements"].aggregate(sensor_pipeline))
    sensor_df = pd.DataFrame([
        {
            "datetime": d["_id"]["date"],
            "pm25_raw": d["pm25_raw"],
            "humidity": d["humidity"],
            "temperature": d["temperature"],
            "pressure": d["pressure"],
            "heat_index": d.get("heat_index"),
        }
        for d in sensor_docs
    ])

    # Hourly reference means
    ref_pipeline = [
        {
            "$group": {
                "_id": {
                    "date": {"$dateTrunc": {"date": "$createdAt", "unit": "hour"}},
                },
                "pm25_raw": {"$avg": "$pm25_raw"},
                "pm25_ugm3": {"$avg": "$pm25_ugm3"},
            }
        },
        {"$sort": {"_id.date": 1}},
    ]
    ref_docs = list(db["referencemeasurements"].aggregate(ref_pipeline))
    ref_df = pd.DataFrame([
        {
            "datetime": d["_id"]["date"],
            "pm25_ref_raw": d["pm25_raw"],
            "pm25_ref_ugm3": d.get("pm25_ugm3"),
        }
        for d in ref_docs
    ])

    # Use pm25_ugm3 if available, else convert AQI
    if "pm25_ref_ugm3" in ref_df.columns:
        ref_df["pm25_ref"] = ref_df["pm25_ref_ugm3"].fillna(
            ref_df["pm25_ref_raw"].apply(aqi_to_ugm3_pm25)
        )
    else:
        ref_df["pm25_ref"] = ref_df["pm25_ref_raw"].apply(aqi_to_ugm3_pm25)

    # Inner join on datetime
    merged = sensor_df.merge(ref_df[["datetime", "pm25_ref"]], on="datetime", how="inner")
    merged["date"] = merged["datetime"].dt.date
    merged["hour"] = merged["datetime"].dt.hour
    merged["date"] = pd.to_datetime(merged["date"])

    logger.info("Loaded %d paired hourly records from MongoDB", len(merged))
    return merged


def prepare_features(df):
    """Ensure all feature columns exist and compute derived ones."""
    df = df.copy()

    if "hour" not in df.columns and "datetime" in df.columns:
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    elif "hour" not in df.columns:
        df["hour"] = df["date"].dt.hour if hasattr(df["date"].dt, "hour") else 0

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek

    if "heat_index" not in df.columns or df["heat_index"].isna().all():
        df["heat_index"] = df.apply(
            lambda r: compute_heat_index(r.get("temperature"), r.get("humidity")), axis=1
        )

    # Drop rows with missing features or target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Sklearn model training ───────────────────────────────────────────────────


def train_sklearn_models(X_train, y_train):
    """Train and return dict of sklearn/xgboost models with timings."""
    models = {}
    tscv = TimeSeriesSplit(n_splits=5)

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
    knn_gs = GridSearchCV(KNeighborsRegressor(), knn_params, cv=tscv, scoring="r2", n_jobs=-1)
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
    svr_gs = GridSearchCV(SVR(), svr_params, cv=tscv, scoring="r2", n_jobs=-1)
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
        rf_params, n_iter=20, cv=tscv, scoring="r2",
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
        xgb_params, n_iter=20, cv=tscv, scoring="r2",
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
        dtr_params, cv=tscv, scoring="r2", n_jobs=-1,
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
    model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


def _create_sliding_windows(X, y, window_size):
    """Create sliding windows from contiguous sequences.

    Gaps in the hourly data break window continuity. Assumes data is sorted
    chronologically and rows are hourly.
    """
    Xw, yw = [], []
    i = 0
    while i + window_size <= len(X):
        Xw.append(X[i:i + window_size])
        yw.append(y[i + window_size - 1])
        i += 1
    if not Xw:
        return np.array([]).reshape(0, window_size, X.shape[1]), np.array([])
    return np.array(Xw), np.array(yw)


def train_lstm_model(X_train_w, y_train_w, X_val_w, y_val_w, input_dim, bidirectional=False):
    """Train LSTM or BiLSTM with early stopping."""
    device = _get_device()
    model = CalibrationLSTM(input_dim=input_dim, bidirectional=bidirectional).to(device)
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
    """Run TimeSeriesSplit CV for all sklearn models.

    Returns dict: model_name -> {"r2": (mean, std), "rmse": ..., "mae": ...}
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}

    for name, (model, _) in models_dict.items():
        r2s, rmses, maes = [], [], []
        for train_idx, val_idx in tscv.split(X):
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
        logger.info(
            "CV %s: R2=%.4f±%.4f  RMSE=%.2f±%.2f  MAE=%.2f±%.2f",
            name, np.mean(r2s), np.std(r2s), np.mean(rmses), np.std(rmses),
            np.mean(maes), np.std(maes),
        )

    return cv_results


def cross_validate_nn(X_scaled, y, model_class, nn_kwargs, window=None):
    """Run TimeSeriesSplit CV for a neural network model.

    If window is set, create sliding windows before training (LSTM/BiLSTM).
    """
    tscv = TimeSeriesSplit(n_splits=5)
    r2s, rmses, maes = [], [], []
    device = _get_device()

    for train_idx, val_idx in tscv.split(X_scaled):
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
                preds = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()
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


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Calibration model selection")
    parser.add_argument("--csv", type=str, help="Path to paired CSV file")
    parser.add_argument("--mongo", action="store_true", help="Force MongoDB mode (real co-location)")
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="Data mode: 'synthetic' (simulation-based) or 'real' (co-location paired data)",
    )
    parser.add_argument("--days", type=int, default=30,
                        help="Limit reference data to last N days (default: 30)")
    args = parser.parse_args()

    for d in [RESULTS_DIR, MODELS_DIR, SCALERS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.csv:
        logger.info("Data mode: REAL (CSV file)")
        df = load_from_csv(args.csv)
    elif args.mongo or args.mode == "real":
        logger.info("Data mode: REAL (co-location paired data)")
        df = load_from_mongodb()
    elif args.mode == "synthetic":
        logger.info("Data mode: SYNTHETIC (simulation-based model selection)")
        ref_df = load_reference_from_mongodb()
        # Filter to last N days
        if args.days > 0:
            cutoff = ref_df["datetime"].max() - pd.Timedelta(days=args.days)
            before = len(ref_df)
            ref_df = ref_df[ref_df["datetime"] >= cutoff].reset_index(drop=True)
            logger.info("Filtered to last %d days: %d -> %d hourly records",
                        args.days, before, len(ref_df))
        df = build_synthetic_dataset(ref_df, random_state=RANDOM_SEED)
        logger.info("Generated %d synthetic sensor readings from reference data", len(df))
    else:
        # Auto-detect: check for CSV in data/
        csvs = globmod.glob(os.path.join(DATA_DIR, "*.csv"))
        if csvs:
            df = load_from_csv(csvs[0])
        else:
            df = load_from_mongodb()

    df = prepare_features(df)
    logger.info("Dataset: %d rows, %d features", len(df), len(FEATURE_COLS))

    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)

    # ── Scale ─────────────────────────────────────────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    joblib.dump(scaler, os.path.join(SCALERS_DIR, "calibration_scaler.joblib"))
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ── Train sklearn models ──────────────────────────────────────────────
    logger.info("Training sklearn/XGBoost models...")
    sklearn_models = train_sklearn_models(X_train, y_train)

    # ── Cross-validate sklearn models ─────────────────────────────────────
    logger.info("Cross-validating sklearn models...")
    cv_results = cross_validate_sklearn(X_train, y_train, sklearn_models)

    # ── Train ANN ─────────────────────────────────────────────────────────
    logger.info("Training ANN...")
    ann_model, ann_time = train_ann(X_train, y_train, X_test, y_test)
    logger.info("ANN trained in %.1fs", ann_time)

    logger.info("Cross-validating ANN...")
    cv_ann = cross_validate_nn(X_train, y_train, CalibrationANN, {"input_dim": X_train.shape[1]})
    cv_results["ANN"] = cv_ann

    # ── Train LSTM ────────────────────────────────────────────────────────
    logger.info("Preparing LSTM sliding windows...")
    X_train_w, y_train_w = _create_sliding_windows(X_train, y_train, LSTM_WINDOW)
    X_test_w, y_test_w = _create_sliding_windows(X_test, y_test, LSTM_WINDOW)

    if len(X_train_w) > 0:
        logger.info("Training LSTM (windows: train=%d, test=%d)...", len(X_train_w), len(X_test_w))
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
        lstm_model = bilstm_model = None
        lstm_time = bilstm_time = 0.0

    # ── Final evaluation on test set ──────────────────────────────────────
    results_rows = []
    all_models = {}
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

    # ANN
    ann_model.eval()
    with torch.no_grad():
        ann_pred = ann_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
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

    # LSTM / BiLSTM
    for name, model, train_time in [("LSTM", lstm_model, lstm_time), ("BiLSTM", bilstm_model, bilstm_time)]:
        if model is None:
            continue
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test_w, dtype=torch.float32).to(device)).cpu().numpy()
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

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "calibration_comparison.csv"), index=False)
    logger.info("Results:\n%s", results_df.to_string(index=False))

    # Determine best model by highest CV R² mean (tiebreak: lowest RMSE mean)
    valid = results_df.dropna(subset=["r2_mean"])
    if len(valid) > 0:
        best_idx = valid.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).index[0]
        best_row = results_df.loc[best_idx]
    else:
        best_row = results_df.loc[results_df["r2_final"].idxmax()]
    best_name = best_row["model"]
    logger.info("Best model: %s (CV R2=%.4f±%.4f, RMSE=%.2f±%.2f)",
                best_name, best_row["r2_mean"], best_row["r2_std"],
                best_row["rmse_mean"], best_row["rmse_std"])

    with open(os.path.join(RESULTS_DIR, "calibration_best_model_name.txt"), "w") as f:
        f.write(best_name)

    # Save best model
    model_type, best_model = all_models[best_name]
    if model_type == "sklearn":
        joblib.dump(best_model, os.path.join(MODELS_DIR, "calibration_best_model.joblib"))
    else:
        torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "calibration_best_model.pt"))
        # Also save full model for easier loading
        torch.save(best_model, os.path.join(MODELS_DIR, "calibration_best_model_full.pt"))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models_names = results_df["model"].tolist()
    x_pos = np.arange(len(models_names))

    axes[0].bar(x_pos, results_df["r2_final"], color="steelblue")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models_names, rotation=45, ha="right")
    axes[0].set_ylabel("R² (test set)")
    axes[0].set_title("Calibration — R² Comparison")

    axes[1].bar(x_pos, results_df["rmse_final"], color="coral")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models_names, rotation=45, ha="right")
    axes[1].set_ylabel("RMSE (test set)")
    axes[1].set_title("Calibration — RMSE Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_comparison.png"), dpi=150)
    plt.close()

    logger.info("Done. Outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Retrain the best forecasting model using all available reference data."""

import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymongo import MongoClient
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BEST_FORECASTING_MODEL,
    FORECAST_HORIZON,
    FORECAST_LOOKBACK,
    MONGODB_URI,
    SELECTION_MODELS_DIR,
    SELECTION_SCALERS_DIR,
)

# Add selection dir for utils and model classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "selection"))
from utils import RANDOM_SEED, aqi_to_ugm3_pm25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10


def get_db():
    client = MongoClient(MONGODB_URI)
    return client["air-pollution"]


def fetch_reference_daily(db):
    """Fetch all reference station data aggregated to daily means."""
    ref_col = db["referencemeasurements"]

    pipeline = [
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                "pm25_ugm3": {"$avg": "$pm25_ugm3"},
                "pm25_raw": {"$avg": "$pm25_raw"},
                "temperature": {"$avg": "$temperature"},
                "humidity": {"$avg": "$humidity"},
                "pressure": {"$avg": "$pressure"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    docs = list(ref_col.aggregate(pipeline))
    if not docs:
        return pd.DataFrame()

    records = []
    for d in docs:
        pm25 = d.get("pm25_ugm3")
        if pm25 is None and d.get("pm25_raw") is not None:
            pm25 = aqi_to_ugm3_pm25(d["pm25_raw"])
        if pm25 is None:
            continue
        records.append({
            "date": pd.Timestamp(d["_id"]),
            "pm25": pm25,
            "temperature": d.get("temperature"),
            "humidity": d.get("humidity"),
            "pressure": d.get("pressure"),
        })

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def build_features(df):
    """Build features matching the selection pipeline."""
    df = df.copy()

    for lag in range(1, FORECAST_LOOKBACK + 1):
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df["pm25"].shift(1).rolling(window).mean()
    df["rolling_std_7"] = df["pm25"].shift(1).rolling(7).std()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    weather_cols = ["temperature", "humidity", "pressure"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)

    for h in range(1, FORECAST_HORIZON + 1):
        df[f"target_day_{h}"] = df["pm25"].shift(-h)

    return df


def build_sequences(X, y, lookback):
    """Build sliding window sequences for sequential models."""
    X_list, y_list = [], []
    for i in range(lookback, len(X)):
        X_list.append(X[i - lookback:i])
        y_list.append(y[i])
    return np.array(X_list), np.array(y_list)


def retrain_xgboost(X_train, y_train):
    """Retrain XGBoost model."""
    base = XGBRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def retrain_torch_model(X_train_seq, y_train_seq):
    """Retrain the best PyTorch model by loading and retraining."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the full model from selection
    pt_path = os.path.join(SELECTION_MODELS_DIR, "forecasting_best_model_full.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")

    model = torch.load(pt_path, map_location=device, weights_only=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Use last 20% as validation for early stopping
    split = int(len(X_train_seq) * 0.8)
    X_val = torch.tensor(X_train_seq[split:], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_train_seq[split:], dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_retrain():
    db = get_db()

    logger.info("Fetching reference data...")
    df = fetch_reference_daily(db)
    if df.empty:
        logger.warning("No reference data available")
        return {"status": "no_data"}

    logger.info("Building features from %d daily records...", len(df))
    df = build_features(df)

    target_cols = [f"target_day_{h}" for h in range(1, FORECAST_HORIZON + 1)]
    exclude = ["date", "pm25"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    df_clean = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    logger.info("Clean dataset: %d rows", len(df_clean))

    if len(df_clean) < 50:
        logger.warning("Insufficient data: %d rows (need >= 50)", len(df_clean))
        return {"status": "insufficient_data", "rows": len(df_clean)}

    X = df_clean[feature_cols].values
    y = df_clean[target_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_name = BEST_FORECASTING_MODEL
    logger.info("Retraining model: %s", model_name)

    # Save dir
    save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "XGBoost":
        model = retrain_xgboost(X_scaled, y)
        joblib.dump(model, os.path.join(save_dir, "forecasting_best_model.joblib"))
        joblib.dump(scaler, os.path.join(save_dir, "forecasting_scaler.joblib"))
    else:
        # Sequential model: build sequences
        X_seq, y_seq = build_sequences(X_scaled, y, FORECAST_LOOKBACK)
        if len(X_seq) < 30:
            logger.warning("Insufficient sequences: %d", len(X_seq))
            return {"status": "insufficient_sequences", "count": len(X_seq)}

        model = retrain_torch_model(X_seq, y_seq)
        torch.save(model.state_dict(), os.path.join(save_dir, "forecasting_best_model.pt"))
        torch.save(model, os.path.join(save_dir, "forecasting_best_model_full.pt"))
        joblib.dump(scaler, os.path.join(save_dir, "forecasting_scaler.joblib"))

    result = {
        "status": "ok",
        "model": model_name,
        "training_samples": len(df_clean),
        "retrained_at": datetime.utcnow().isoformat(),
    }
    logger.info("Retrain complete: %s", result)
    return result


if __name__ == "__main__":
    result = run_retrain()
    print(json.dumps(result, default=str))

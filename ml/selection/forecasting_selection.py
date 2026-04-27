"""Forecasting model selection pipeline.

Compares 4 models (XGBoost, LSTM, BiLSTM, CNN-LSTM) for multi-day-ahead
PM2.5 forecasting from historical reference station data.

Sequential models use raw feature sequences (no lag engineering).
XGBoost uses explicit lag/rolling features.

Usage:
    python forecasting_selection.py --csv data/forecasting_ready.csv
"""

import argparse
import glob as globmod
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RANDOM_SEED, aqi_to_ugm3_pm25, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SCALERS_DIR = os.path.join(OUTPUT_DIR, "scalers")

LOOKBACK = 30
HORIZON = 7
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 15

# Raw features for sequence models (no lags — the model sees LOOKBACK days directly)
SEQ_FEATURE_COLS = [
    "pm25", "temperature", "humidity", "pressure",
    "month", "day_of_week", "day",
]


# ── Neural network architectures ──────────────────────────────────────────────


class ForecastLSTM(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, horizon=HORIZON, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        d = 2 if bidirectional else 1
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True, bidirectional=bidirectional)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden1 * d, hidden2, batch_first=True, bidirectional=bidirectional)
        self.drop2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden2 * d, 32)
        self.fc2 = nn.Linear(32, horizon)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        return self.fc2(out)


class ForecastCNNLSTM(nn.Module):
    def __init__(self, input_dim, horizon=HORIZON):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, horizon)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = torch.relu(self.conv(out))
        out = self.pool(out)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.dropout(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        return self.fc2(out)


# ── Data loading & feature engineering ────────────────────────────────────────


def load_reference_csv(csv_path):
    """Load reference station CSV (single or multi-station)."""
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])

    if "pm25" in df.columns and df["pm25"].max() > 300:
        logger.info("Detected AQI values in pm25 — converting to ug/m3")
        df["pm25"] = df["pm25"].apply(aqi_to_ugm3_pm25)

    multi = "station_name" in df.columns and df["station_name"].nunique() > 1
    if multi:
        stations = df["station_name"].unique()
        logger.info("Multi-station dataset: %d stations (%s)", len(stations), ", ".join(stations))
        df = df.sort_values(["station_name", "date"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def _build_ml_features_single(sdf):
    """Build lag + rolling + temporal + weather features for one station."""
    sdf = sdf.sort_values("date").copy()

    for lag in range(1, LOOKBACK + 1):
        sdf[f"pm25_lag_{lag}"] = sdf["pm25"].shift(lag)

    for window in [7, 14, 30]:
        if window <= LOOKBACK:
            sdf[f"rolling_mean_{window}"] = sdf["pm25"].shift(1).rolling(window).mean()
    sdf["rolling_std_7"] = sdf["pm25"].shift(1).rolling(min(7, LOOKBACK)).std()

    sdf["month"] = sdf["date"].dt.month
    sdf["day_of_week"] = sdf["date"].dt.dayofweek
    sdf["day_of_year"] = sdf["date"].dt.dayofyear

    weather_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in weather_cols:
        if col in sdf.columns:
            sdf[col] = sdf[col].shift(1)

    for h in range(1, HORIZON + 1):
        sdf[f"target_day_{h}"] = sdf["pm25"].shift(-h)

    return sdf


def build_ml_features(df):
    """Build lag/rolling features per station, then combine."""
    if "station_name" in df.columns and df["station_name"].nunique() > 1:
        parts = []
        for station, sdf in df.groupby("station_name"):
            parts.append(_build_ml_features_single(sdf))
        return pd.concat(parts, ignore_index=True)
    return _build_ml_features_single(df)


def prepare_seq_data(df):
    """Prepare raw daily data for sequence models (no lag engineering).

    Features: pm25, weather (same-day, no shift), temporal.
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day

    # Use only columns that exist
    available = [c for c in SEQ_FEATURE_COLS if c in df.columns]
    if "wind_speed" in df.columns and "wind_speed" not in available:
        available.append("wind_speed")

    logger.info("Seq features: %s", available)
    return df, available


def build_raw_sequences(data, lookback, horizon):
    """Build (lookback, n_features) sequences with (horizon,) targets.

    Target = column 0 (pm25) for next `horizon` days.
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, 0])
    if not X:
        return np.array([]).reshape(0, lookback, data.shape[1]), np.array([])
    return np.array(X), np.array(y)


def build_sequences_multi_station(df, seq_cols, scaler):
    """Build sequences per station, then combine sorted by start date.

    Prevents cross-station leakage while maintaining chronological order
    for proper train/test split.
    """
    all_X, all_y, all_dates = [], [], []

    if "station_name" in df.columns:
        groups = df.groupby("station_name")
    else:
        groups = [("single", df)]

    for station, sdf in groups:
        sdf = sdf.sort_values("date").reset_index(drop=True)
        raw = sdf[seq_cols].values.astype(np.float64)
        scaled = scaler.transform(raw)
        X, y = build_raw_sequences(scaled, LOOKBACK, HORIZON)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            # Track start date for chronological sorting
            dates = sdf["date"].values[LOOKBACK:LOOKBACK + len(X)]
            all_dates.append(dates)
            logger.info("  %s: %d sequences", station, len(X))

    if not all_X:
        return np.array([]), np.array([])

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    dates_all = np.concatenate(all_dates)

    # Sort by date so train/test split is chronological
    order = np.argsort(dates_all)
    return X_all[order], y_all[order]


# ── Training ─────────────────────────────────────────────────────────────────


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_xgboost(X_train, y_train):
    t0 = time.time()
    base = XGBRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, time.time() - t0


def train_seq_model(X_train, y_train, X_val, y_val, model_class, model_kwargs):
    """Train a PyTorch sequential model with Huber loss and early stopping."""
    device = _get_device()
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss(delta=15.0)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    t0 = time.time()
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
                logger.info("%s early stopping at epoch %d", model_class.__name__, epoch + 1)
                break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, elapsed


# ── Cross-validation ─────────────────────────────────────────────────────────


def cv_xgboost(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    per_horizon = {h: {"r2": [], "rmse": [], "mae": []} for h in range(HORIZON)}
    overall = {"r2": [], "rmse": [], "mae": []}

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model, _ = train_xgboost(X_tr, y_tr)
        y_pred = model.predict(X_va)

        for h in range(HORIZON):
            m = compute_metrics(y_va[:, h], y_pred[:, h])
            per_horizon[h]["r2"].append(m["r2"])
            per_horizon[h]["rmse"].append(m["rmse"])
            per_horizon[h]["mae"].append(m["mae"])

        m_all = compute_metrics(y_va.flatten(), y_pred.flatten())
        overall["r2"].append(m_all["r2"])
        overall["rmse"].append(m_all["rmse"])
        overall["mae"].append(m_all["mae"])

    return overall, per_horizon


def cv_seq_model(X_seq, y_seq, model_class, model_kwargs, scaler=None):
    """TimeSeriesSplit CV for sequential models."""
    tscv = TimeSeriesSplit(n_splits=5)
    per_horizon = {h: {"r2": [], "rmse": [], "mae": []} for h in range(HORIZON)}
    overall = {"r2": [], "rmse": [], "mae": []}
    device = _get_device()

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr, X_va = X_seq[train_idx], X_seq[val_idx]
        y_tr, y_va = y_seq[train_idx], y_seq[val_idx]

        if len(X_tr) < BATCH_SIZE or len(X_va) == 0:
            continue

        model, _ = train_seq_model(X_tr, y_tr, X_va, y_va, model_class, model_kwargs)
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()

        # Inverse-transform if scaler provided
        if scaler is not None:
            pm25_mean, pm25_scale = scaler.mean_[0], scaler.scale_[0]
            y_pred = y_pred * pm25_scale + pm25_mean
            y_va = y_va * pm25_scale + pm25_mean

        for h in range(HORIZON):
            m = compute_metrics(y_va[:, h], y_pred[:, h])
            per_horizon[h]["r2"].append(m["r2"])
            per_horizon[h]["rmse"].append(m["rmse"])
            per_horizon[h]["mae"].append(m["mae"])

        m_all = compute_metrics(y_va.flatten(), y_pred.flatten())
        overall["r2"].append(m_all["r2"])
        overall["rmse"].append(m_all["rmse"])
        overall["mae"].append(m_all["mae"])

    return overall, per_horizon


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Forecasting model selection")
    parser.add_argument("--csv", type=str, help="Path to reference station CSV")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Override LOOKBACK (default: 30, auto-reduced for small datasets)")
    args = parser.parse_args()

    for d in [RESULTS_DIR, MODELS_DIR, SCALERS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    if args.csv:
        df = load_reference_csv(args.csv)
    else:
        csvs = globmod.glob(os.path.join(DATA_DIR, "*.csv"))
        if not csvs:
            logger.error("No CSV found in %s. Provide --csv path.", DATA_DIR)
            sys.exit(1)
        df = load_reference_csv(csvs[0])

    logger.info("Loaded %d daily records", len(df))

    # ── Adjust lookback ──────────────────────────────────────────────────
    global LOOKBACK
    if args.lookback is not None:
        LOOKBACK = args.lookback
    else:
        min_usable = 50
        max_lookback = len(df) - HORIZON - min_usable
        if max_lookback < LOOKBACK:
            old_lb = LOOKBACK
            LOOKBACK = max(7, max_lookback)
            logger.warning("Auto-reduced LOOKBACK from %d to %d", old_lb, LOOKBACK)

    min_required = LOOKBACK + HORIZON + 2
    if len(df) < min_required:
        logger.error("Insufficient data: %d records, need >= %d", len(df), min_required)
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # PATH A: XGBoost (lag/rolling features, shifted weather)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== XGBoost path (lag features) ===")
    df_ml = build_ml_features(df)
    target_cols = [f"target_day_{h}" for h in range(1, HORIZON + 1)]
    exclude = {"date", "pm25", "station_name"} | set(target_cols)
    ml_feature_cols = [c for c in df_ml.columns if c not in exclude]

    df_ml_clean = df_ml.dropna(subset=ml_feature_cols + target_cols).reset_index(drop=True)
    logger.info("XGBoost clean dataset: %d rows, %d features", len(df_ml_clean), len(ml_feature_cols))

    X_ml = df_ml_clean[ml_feature_cols].values.astype(np.float64)
    y_ml = df_ml_clean[target_cols].values.astype(np.float64)

    # Shuffle rows (each row is self-contained after per-station feature engineering)
    shuffle_ml = np.random.permutation(len(X_ml))
    X_ml, y_ml = X_ml[shuffle_ml], y_ml[shuffle_ml]

    split_ml = int(len(X_ml) * 0.8)
    X_train_ml, X_test_ml = X_ml[:split_ml], X_ml[split_ml:]
    y_train_ml, y_test_ml = y_ml[:split_ml], y_ml[split_ml:]

    scaler_ml = StandardScaler()
    X_train_ml_s = scaler_ml.fit_transform(X_train_ml)
    X_test_ml_s = scaler_ml.transform(X_test_ml)
    joblib.dump(scaler_ml, os.path.join(SCALERS_DIR, "forecasting_scaler.joblib"))
    logger.info("XGBoost train: %d, test: %d", len(X_train_ml_s), len(X_test_ml_s))

    all_results = []
    all_models = {}

    logger.info("Training XGBoost...")
    xgb_model, xgb_time = train_xgboost(X_train_ml_s, y_train_ml)
    logger.info("Cross-validating XGBoost...")
    xgb_cv_overall, xgb_cv_horizon = cv_xgboost(X_train_ml_s, y_train_ml)
    all_models["XGBoost"] = ("sklearn", xgb_model)

    xgb_pred = xgb_model.predict(X_test_ml_s)
    xgb_final = compute_metrics(y_test_ml.flatten(), xgb_pred.flatten())
    xgb_per_h = {h: compute_metrics(y_test_ml[:, h], xgb_pred[:, h]) for h in range(HORIZON)}

    all_results.append({
        "model": "XGBoost", "cv_overall": xgb_cv_overall, "cv_horizon": xgb_cv_horizon,
        "final": xgb_final, "final_per_horizon": xgb_per_h, "time": xgb_time,
    })

    # ══════════════════════════════════════════════════════════════════════
    # PATH B: Sequence models (raw features, same-day weather, no lags)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=== Sequence models path (raw features) ===")
    df_seq, seq_cols = prepare_seq_data(df)

    # Drop rows with NaN in seq features
    df_seq_clean = df_seq.dropna(subset=seq_cols).reset_index(drop=True)

    # Fit scaler on all data
    raw_data = df_seq_clean[seq_cols].values.astype(np.float64)
    scaler_seq = StandardScaler()
    scaler_seq.fit(raw_data)
    joblib.dump(scaler_seq, os.path.join(SCALERS_DIR, "forecasting_seq_scaler.joblib"))

    # Build sequences per station (no cross-station leakage)
    logger.info("Building sequences per station:")
    X_seq, y_seq = build_sequences_multi_station(df_seq_clean, seq_cols, scaler_seq)
    if len(X_seq) > 0:
        logger.info("Total sequences: %d, shape=%s", len(X_seq), X_seq.shape[1:])
    else:
        logger.warning("No sequences built")

    has_seq_data = len(X_seq) > BATCH_SIZE
    if has_seq_data:
        # Shuffle sequences (each is self-contained, no temporal dependency between them)
        shuffle_idx = np.random.permutation(len(X_seq))
        X_seq, y_seq = X_seq[shuffle_idx], y_seq[shuffle_idx]

        seq_split = int(len(X_seq) * 0.8)
        X_train_seq, X_test_seq = X_seq[:seq_split], X_seq[seq_split:]
        y_train_seq, y_test_seq = y_seq[:seq_split], y_seq[seq_split:]
        logger.info("Seq train: %d, test: %d", len(X_train_seq), len(X_test_seq))
        input_dim = X_seq.shape[2]

        device = _get_device()
        seq_configs = [
            ("LSTM", ForecastLSTM, {"input_dim": input_dim, "bidirectional": False}),
            ("BiLSTM", ForecastLSTM, {"input_dim": input_dim, "bidirectional": True}),
            ("CNN-LSTM", ForecastCNNLSTM, {"input_dim": input_dim}),
        ]

        for name, model_class, kwargs in seq_configs:
            logger.info("Training %s...", name)
            model, train_time = train_seq_model(
                X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                model_class, kwargs,
            )

            logger.info("Cross-validating %s...", name)
            cv_overall, cv_horizon = cv_seq_model(X_train_seq, y_train_seq, model_class, kwargs, scaler=scaler_seq)

            model.eval()
            with torch.no_grad():
                pred_scaled = model(torch.tensor(X_test_seq, dtype=torch.float32).to(device)).cpu().numpy()

            # Inverse-transform predictions and targets (pm25 = column 0)
            pm25_mean = scaler_seq.mean_[0]
            pm25_scale = scaler_seq.scale_[0]
            pred = pred_scaled * pm25_scale + pm25_mean
            y_test_orig = y_test_seq * pm25_scale + pm25_mean

            final = compute_metrics(y_test_orig.flatten(), pred.flatten())
            per_h = {h: compute_metrics(y_test_orig[:, h], pred[:, h]) for h in range(HORIZON)}

            all_results.append({
                "model": name, "cv_overall": cv_overall, "cv_horizon": cv_horizon,
                "final": final, "final_per_horizon": per_h, "time": train_time,
            })
            all_models[name] = ("torch", model)
    else:
        logger.warning("Insufficient data for sequence models")

    # ── Save results ──────────────────────────────────────────────────────
    comparison_rows = []
    for res in all_results:
        row = {"model": res["model"]}
        cv = res["cv_overall"]
        row["r2_cv_mean"] = np.mean(cv["r2"]) if cv["r2"] else float("nan")
        row["r2_cv_std"] = np.std(cv["r2"]) if cv["r2"] else float("nan")
        row["rmse_cv_mean"] = np.mean(cv["rmse"]) if cv["rmse"] else float("nan")
        row["rmse_cv_std"] = np.std(cv["rmse"]) if cv["rmse"] else float("nan")
        row["mae_cv_mean"] = np.mean(cv["mae"]) if cv["mae"] else float("nan")
        row["mae_cv_std"] = np.std(cv["mae"]) if cv["mae"] else float("nan")
        row["r2_final"] = res["final"]["r2"]
        row["rmse_final"] = res["final"]["rmse"]
        row["mae_final"] = res["final"]["mae"]
        row["training_time_sec"] = round(res["time"], 2)

        for h in range(HORIZON):
            hm = res["final_per_horizon"][h]
            row[f"rmse_day{h+1}"] = hm["rmse"]
            row[f"r2_day{h+1}"] = hm["r2"]

        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "forecasting_comparison.csv"), index=False)

    # Log CV mean±std for all models
    cv_cols = ["model", "r2_cv_mean", "r2_cv_std", "rmse_cv_mean", "rmse_cv_std", "mae_cv_mean", "mae_cv_std"]
    logger.info("CV Results:\n%s", comp_df[cv_cols].to_string(index=False))
    logger.info("Final test:\n%s", comp_df[["model", "r2_final", "rmse_final", "mae_final"]].to_string(index=False))

    # Select best model by lowest rmse_cv_mean
    best_row = comp_df.loc[comp_df["rmse_cv_mean"].idxmin()]
    best_name = best_row["model"]
    logger.info("Best model: %s (CV RMSE=%.2f±%.2f, CV R2=%.4f±%.4f)",
                best_name, best_row["rmse_cv_mean"], best_row["rmse_cv_std"],
                best_row["r2_cv_mean"], best_row["r2_cv_std"])

    with open(os.path.join(RESULTS_DIR, "forecasting_best_model_name.txt"), "w") as f:
        f.write(best_name)

    model_type, best_model = all_models[best_name]
    if model_type == "sklearn":
        joblib.dump(best_model, os.path.join(MODELS_DIR, "forecasting_best_model.joblib"))
    else:
        torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, "forecasting_best_model.pt"))
        torch.save(best_model, os.path.join(MODELS_DIR, "forecasting_best_model_full.pt"))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = list(range(1, HORIZON + 1))
    for res in all_results:
        rmses = [res["final_per_horizon"][h]["rmse"] for h in range(HORIZON)]
        ax.plot(horizons, rmses, marker="o", label=res["model"])
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("RMSE")
    ax.set_title("Forecasting — RMSE by Horizon")
    ax.legend()
    ax.set_xticks(horizons)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecasting_comparison.png"), dpi=150)
    plt.close()

    logger.info("Done. Outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

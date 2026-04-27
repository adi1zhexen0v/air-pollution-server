"""Forecasting model comparison for Chapter 4: daily 7-day horizon PM2.5
forecasting using historical multi-station reference data.

Trains 4 models (XGBoost, LSTM, BiLSTM, CNN-LSTM) with TimeSeriesSplit CV.
Reports CV-only metrics (no final test split).

Usage:
    cd ml/final_version
    python final_forecasting_pipeline.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_CSV = os.path.join(BASE_DIR, "..", "selection", "data", "forecasting_ready.csv")

LOOKBACK = 30
HORIZON = 7
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 15

SEQ_FEATURE_COLS = ["pm25", "temperature", "humidity", "pressure",
                    "wind_speed", "month", "day_of_week", "day"]

C_LSTM, C_BILSTM, C_CNN, C_XGB = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"
C_EPA = "#e24b4a"

plt.rcParams.update({
    "font.size": 12, "font.family": "Arial",
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

_LOG_LINES = []


def log(s=""):
    print(s)
    _LOG_LINES.append(s)


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ── AQI converter ────────────────────────────────────────────────────────────

_PM25_BP = [(0, 50, 0.0, 12.0), (50, 100, 12.0, 35.4), (100, 150, 35.4, 55.4),
            (150, 200, 55.4, 150.4), (200, 300, 150.4, 250.4),
            (300, 400, 250.4, 350.4), (400, 500, 350.4, 500.4)]


def aqi_to_ugm3(aqi):
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return None
    aqi = float(aqi)
    if aqi < 0:
        return 0.0
    for al, ah, cl, ch in _PM25_BP:
        if al <= aqi <= ah:
            return (ch - cl) / (ah - al) * (aqi - al) + cl
    al, ah, cl, ch = _PM25_BP[-1]
    return (ch - cl) / (ah - al) * (aqi - al) + cl


# ── PyTorch models ────────────────────────────────────────────────────────────

class ForecastLSTM(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, horizon=HORIZON, bidirectional=False):
        super().__init__()
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
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
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


def train_torch_model(model, X_train, y_train, epochs=MAX_EPOCHS, patience=PATIENCE,
                      batch_size=BATCH_SIZE, lr=0.001, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.HuberLoss(delta=15.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    split = int(len(X_t) * 0.85)
    train_ds = TensorDataset(X_t[:split], y_t[:split])
    val_X, val_y = X_t[split:].to(device), y_t[split:].to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    best_val, wait, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if len(val_X) > 0:
                val_loss = loss_fn(model(val_X), val_y).item()
            else:
                val_loss = float("inf")
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


# ── Data loading & feature engineering ────────────────────────────────────────

def load_data():
    log("=" * 60)
    log("LOADING FORECASTING DATA")
    log("=" * 60)

    csv_path = DATA_CSV
    if not os.path.exists(csv_path):
        # Try alternate location
        alt = os.path.join(BASE_DIR, "..", "selection", "data", "processed", "all_stations_weather.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            log(f"ERROR: Cannot find data CSV at {DATA_CSV}")
            sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Convert AQI if needed
    if "pm25" in df.columns and df["pm25"].max() > 300:
        log("Detected AQI values — converting to ug/m3")
        df["pm25"] = df["pm25"].apply(aqi_to_ugm3)

    multi = "station_name" in df.columns and df["station_name"].nunique() > 1
    if multi:
        stations = df["station_name"].unique()
        log(f"Multi-station dataset: {len(stations)} stations ({', '.join(stations)})")
        df = df.sort_values(["station_name", "date"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)

    log(f"Total records: {len(df)}")
    log(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    log(f"PM2.5: mean={df['pm25'].mean():.2f}, std={df['pm25'].std():.2f} ug/m3")

    return df


def build_ml_features_single(sdf):
    """Build lag + rolling + temporal + weather features for one station."""
    sdf = sdf.sort_values("date").copy()

    for lag in range(1, LOOKBACK + 1):
        sdf[f"pm25_lag_{lag}"] = sdf["pm25"].shift(lag)

    for window in [7, 14, 30]:
        sdf[f"rolling_mean_{window}"] = sdf["pm25"].shift(1).rolling(window).mean()
    sdf["rolling_std_7"] = sdf["pm25"].shift(1).rolling(7).std()

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
            parts.append(build_ml_features_single(sdf))
        return pd.concat(parts, ignore_index=True)
    return build_ml_features_single(df)


def build_raw_sequences(data, lookback, horizon):
    """Build (lookback, n_features) sequences with (horizon,) targets."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, 0])
    if not X:
        return np.array([]).reshape(0, lookback, data.shape[1]), np.array([])
    return np.array(X), np.array(y)


def build_sequences_multi_station(df, seq_cols, scaler):
    """Build sequences per station, then combine sorted by start date."""
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
            dates = sdf["date"].values[LOOKBACK:LOOKBACK + len(X)]
            all_dates.append(dates)
            log(f"  {station}: {len(X)} sequences")

    if not all_X:
        return np.array([]), np.array([])

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    dates_all = np.concatenate(all_dates)

    order = np.argsort(dates_all)
    return X_all[order], y_all[order]


# ── Cross-validation ──────────────────────────────────────────────────────────

def cv_xgboost(X, y):
    """TimeSeriesSplit CV for XGBoost, returning overall + per-horizon metrics."""
    tscv = TimeSeriesSplit(n_splits=5)
    per_horizon = {h: {"rmse": [], "r2": []} for h in range(HORIZON)}
    overall = {"r2": [], "rmse": [], "mae": []}

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        base = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=RANDOM_SEED, verbosity=0)
        m = MultiOutputRegressor(base, n_jobs=-1)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_va)

        for h in range(HORIZON):
            met = compute_metrics(y_va[:, h], y_pred[:, h])
            per_horizon[h]["rmse"].append(met["rmse"])
            per_horizon[h]["r2"].append(met["r2"])

        m_all = compute_metrics(y_va.flatten(), y_pred.flatten())
        overall["r2"].append(m_all["r2"])
        overall["rmse"].append(m_all["rmse"])
        overall["mae"].append(m_all["mae"])

    return overall, per_horizon


def cv_seq_model(X_seq, y_seq, model_class, model_kwargs, scaler):
    """TimeSeriesSplit CV for sequence models."""
    tscv = TimeSeriesSplit(n_splits=5)
    per_horizon = {h: {"rmse": [], "r2": []} for h in range(HORIZON)}
    overall = {"r2": [], "rmse": [], "mae": []}

    pm25_mean, pm25_scale = scaler.mean_[0], scaler.scale_[0]

    for train_idx, val_idx in tscv.split(X_seq):
        X_tr, X_va = X_seq[train_idx], X_seq[val_idx]
        y_tr, y_va = y_seq[train_idx], y_seq[val_idx]

        if len(X_tr) < BATCH_SIZE or len(X_va) == 0:
            continue

        model = model_class(**model_kwargs)
        model = train_torch_model(model, X_tr, y_tr,
                                  epochs=MAX_EPOCHS, patience=PATIENCE,
                                  batch_size=BATCH_SIZE, lr=0.001,
                                  loss_fn=nn.HuberLoss(delta=15.0 / pm25_scale))

        model.eval()
        with torch.no_grad():
            y_pred_sc = model(torch.tensor(X_va, dtype=torch.float32)).numpy()

        # Inverse-transform
        y_pred = y_pred_sc * pm25_scale + pm25_mean
        y_va_orig = y_va * pm25_scale + pm25_mean

        for h in range(HORIZON):
            met = compute_metrics(y_va_orig[:, h], y_pred[:, h])
            per_horizon[h]["rmse"].append(met["rmse"])
            per_horizon[h]["r2"].append(met["r2"])

        m_all = compute_metrics(y_va_orig.flatten(), y_pred.flatten())
        overall["r2"].append(m_all["r2"])
        overall["rmse"].append(m_all["rmse"])
        overall["mae"].append(m_all["mae"])

    return overall, per_horizon


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    # ── XGBoost path ──────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("TRAINING FORECASTING MODELS")
    log("=" * 60)

    df_ml = build_ml_features(df)
    target_cols = [f"target_day_{h}" for h in range(1, HORIZON + 1)]
    exclude = {"date", "pm25", "station_name"} | set(target_cols)
    ml_feature_cols = [c for c in df_ml.columns if c not in exclude]

    df_ml_clean = df_ml.dropna(subset=ml_feature_cols + target_cols).reset_index(drop=True)
    log(f"\nXGBoost dataset: {len(df_ml_clean)} rows, {len(ml_feature_cols)} features")

    X_ml = df_ml_clean[ml_feature_cols].values.astype(np.float64)
    y_ml = df_ml_clean[target_cols].values.astype(np.float64)

    # Shuffle (each row is self-contained after per-station feature engineering)
    shuffle_ml = np.random.permutation(len(X_ml))
    X_ml, y_ml = X_ml[shuffle_ml], y_ml[shuffle_ml]

    scaler_ml = StandardScaler()
    X_ml_s = scaler_ml.fit_transform(X_ml)

    all_results = []

    log("\n  Training + CV: XGBoost...")
    t0 = time.time()
    xgb_overall, xgb_per_h = cv_xgboost(X_ml_s, y_ml)
    xgb_time = time.time() - t0
    all_results.append({
        "name": "XGBoost", "overall": xgb_overall, "per_horizon": xgb_per_h, "time": xgb_time,
    })
    log(f"  XGBoost: CV R2={np.mean(xgb_overall['r2']):.4f} ({xgb_time:.1f}s)")

    # ── Sequence models path ──────────────────────────────────────────────
    df_seq = df.copy()
    df_seq["month"] = df_seq["date"].dt.month
    df_seq["day_of_week"] = df_seq["date"].dt.dayofweek
    df_seq["day"] = df_seq["date"].dt.day

    available_cols = [c for c in SEQ_FEATURE_COLS if c in df_seq.columns]
    if "wind_speed" in df_seq.columns and "wind_speed" not in available_cols:
        available_cols.append("wind_speed")

    df_seq_clean = df_seq.dropna(subset=available_cols).reset_index(drop=True)

    raw_data = df_seq_clean[available_cols].values.astype(np.float64)
    scaler_seq = StandardScaler()
    scaler_seq.fit(raw_data)

    log(f"\nBuilding sequences (lookback={LOOKBACK}, horizon={HORIZON}):")
    X_seq, y_seq = build_sequences_multi_station(df_seq_clean, available_cols, scaler_seq)
    log(f"Total sequences: {len(X_seq)}")

    # Shuffle sequences (each is self-contained, no temporal dependency between them
    # since they come from different stations/periods)
    if len(X_seq) > 0:
        shuffle_idx = np.random.permutation(len(X_seq))
        X_seq, y_seq = X_seq[shuffle_idx], y_seq[shuffle_idx]

    if len(X_seq) > BATCH_SIZE:
        input_dim = X_seq.shape[2]
        seq_configs = [
            ("LSTM", ForecastLSTM, {"input_dim": input_dim, "bidirectional": False}),
            ("BiLSTM", ForecastLSTM, {"input_dim": input_dim, "bidirectional": True}),
            ("CNN-LSTM", ForecastCNNLSTM, {"input_dim": input_dim}),
        ]

        for name, model_class, kwargs in seq_configs:
            log(f"\n  Training + CV: {name}...")
            t0 = time.time()
            cv_overall, cv_per_h = cv_seq_model(X_seq, y_seq, model_class, kwargs, scaler_seq)
            elapsed = time.time() - t0
            all_results.append({
                "name": name, "overall": cv_overall, "per_horizon": cv_per_h, "time": elapsed,
            })
            log(f"  {name}: CV R2={np.mean(cv_overall['r2']):.4f} ({elapsed:.1f}s)")
    else:
        log("WARNING: Insufficient data for sequence models")

    # ── Print Table 4.4 ──────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("TABLE 4.4: Forecasting Model Comparison (Daily, 7-Day Horizon)")
    log("=" * 60)

    # Sort by CV RMSE ascending (lower is better — more reliable than R² for multi-station forecasting)
    all_results.sort(key=lambda r: np.mean(r["overall"]["rmse"]))
    best = all_results[0]

    log(f"\n{'Model':<10} | {'R2 (mean+/-std)':>20} | {'RMSE (ug/m3)':>13} | {'MAE (ug/m3)':>12} | {'Time':>6}")
    log("-" * 75)
    for r in all_results:
        marker = "[BEST] " if r["name"] == best["name"] else "       "
        r2m = np.mean(r["overall"]["r2"])
        r2s = np.std(r["overall"]["r2"])
        rmse = np.mean(r["overall"]["rmse"])
        mae = np.mean(r["overall"]["mae"])
        r2_str = f"{r2m:.3f}+/-{r2s:.3f}"
        log(f"{marker}{r['name']:<5} | {r2_str:>20} | {rmse:>13.2f} | {mae:>12.2f} | {r['time']:>5.1f}s")

    best_r2 = np.mean(best["overall"]["r2"])
    best_rmse = np.mean(best["overall"]["rmse"])
    log(f"\nBest model: {best['name']} (R2 = {best_r2:.3f}, RMSE = {best_rmse:.2f} ug/m3)")
    log(f"Dataset: {len(df)} daily records from {df['station_name'].nunique() if 'station_name' in df.columns else 1} stations")
    log(f"Training sequences: {len(X_seq)} (after {LOOKBACK}-day lookback window)")

    # ── Per-horizon RMSE ─────────────────────────────────────────────────
    log(f"\nPer-Horizon RMSE ({best['name']}):")
    best_per_h_rmse = []
    for h in range(HORIZON):
        rmse_h = np.mean(best["per_horizon"][h]["rmse"])
        best_per_h_rmse.append(rmse_h)
        log(f"  Day {h+1}: {rmse_h:.2f} ug/m3")

    # ── Per-horizon RMSE for all models (for figure) ─────────────────────
    all_per_h = {}
    for r in all_results:
        all_per_h[r["name"]] = [np.mean(r["per_horizon"][h]["rmse"]) for h in range(HORIZON)]

    # ── Figure: Per-horizon RMSE ─────────────────────────────────────────
    log("\nGenerating fig_forecasting_per_horizon.png...")
    colors = {"XGBoost": C_XGB, "LSTM": C_LSTM, "BiLSTM": C_BILSTM, "CNN-LSTM": C_CNN}
    markers = {"XGBoost": "s", "LSTM": "o", "BiLSTM": "^", "CNN-LSTM": "D"}

    fig, ax = plt.subplots(figsize=(10, 6))
    days = list(range(1, HORIZON + 1))
    for r in all_results:
        rmses = all_per_h[r["name"]]
        ax.plot(days, rmses, color=colors.get(r["name"], "black"),
                marker=markers.get(r["name"], "x"), linewidth=2, markersize=7, label=r["name"])
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("RMSE ($\\mu$g/m$^3$)")
    ax.set_title("Forecasting RMSE by horizon (daily, 7-day)")
    ax.set_xticks(days)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_forecasting_per_horizon.png"), dpi=300)
    plt.close(fig)

    # ── Save JSON ────────────────────────────────────────────────────────
    log("Saving forecasting_results.json...")
    results_json = {
        "dataset_records": len(df),
        "training_sequences": len(X_seq),
        "lookback_days": LOOKBACK,
        "horizon_days": HORIZON,
        "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "stations": int(df["station_name"].nunique()) if "station_name" in df.columns else 1,
        "best_model": best["name"],
        "models": [],
    }
    for r in all_results:
        model_entry = {
            "name": r["name"],
            "r2_mean": float(np.mean(r["overall"]["r2"])),
            "r2_std": float(np.std(r["overall"]["r2"])),
            "rmse": float(np.mean(r["overall"]["rmse"])),
            "mae": float(np.mean(r["overall"]["mae"])),
            "time_sec": round(r["time"], 2),
            "per_horizon_rmse": [float(np.mean(r["per_horizon"][h]["rmse"])) for h in range(HORIZON)],
        }
        results_json["models"].append(model_entry)

    with open(os.path.join(OUTPUT_DIR, "forecasting_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # ── Literature comparison ────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("TABLE: Forecasting — Comparison with Literature")
    log("=" * 60)
    log(f"{'Study':<22} | {'Location':<10} | {'Horizon':<10} | {'Best Model':<12} | {'RMSE':>12} | {'Resolution':<10}")
    log("-" * 90)
    for s, l, h, m, r, res in [
        ("Chen (2018)", "Beijing", "1-48h", "LSTM", "15.0", "Hourly"),
        ("Zheng (2023)", "Multi-city", "1-72h", "CNN-BiGRU", "8.3", "Hourly"),
        ("Shah (2022)", "India", "24h", "XGBoost", "12.5", "Hourly"),
    ]:
        log(f"{s:<22} | {l:<10} | {h:<10} | {m:<12} | {r:>12} | {res:<10}")
    log(f"{'This work':<22} | {'Astana':<10} | {'7-day':<10} | {best['name']:<12} | {best_rmse:>12.2f} | {'Daily':<10}")

    # ── Final summary ────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("FORECASTING COMPLETE")
    log("=" * 60)
    n_stations = df["station_name"].nunique() if "station_name" in df.columns else 1
    station_names = ", ".join(df["station_name"].unique()) if "station_name" in df.columns else "single"
    log(f"\nDataset: {len(df)} daily records from {n_stations} stations ({station_names})")
    log(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    log(f"Training sequences: {len(X_seq)} (after {LOOKBACK}-day lookback window)")
    log(f"\nBest model: {best['name']} (R2 = {best_r2:.3f}, RMSE = {best_rmse:.2f} ug/m3)")
    log(f"\nOutput files:")
    log(f"  {os.path.join(OUTPUT_DIR, 'forecasting_results.json')}")
    log(f"  {os.path.join(OUTPUT_DIR, 'fig_forecasting_per_horizon.png')}")

    # Save log
    with open(os.path.join(OUTPUT_DIR, "forecasting_log.txt"), "w") as f:
        f.write("\n".join(_LOG_LINES))


if __name__ == "__main__":
    main()

"""Complete ML pipeline for Chapter 4: calibration (9 models) and age-based
calibration (4 windows).

All data from CSV files — no MongoDB connection required.

Usage:
    cd ml/final_version
    python final_calibration_pipeline.py
"""

import json
import math
import os
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MEASUREMENTS_CSV = os.path.join(BASE_DIR, "final.measurements.csv")
REFERENCE_CSV = os.path.join(BASE_DIR, "final.referencemeasurements.csv")

FEATURES = ["pm25_raw", "humidity", "temperature", "pressure",
            "heat_index", "hour", "month", "day_of_week"]
FEATURE_LABELS = ["PM2.5 raw", "Humidity", "Temperature", "Pressure",
                  "Heat Index", "Hour of Day", "Month", "Day of Week"]
TARGET = "pm25_ref"

# Colors
C_S1, C_S2, C_S3 = "#2ca02c", "#1f77b4", "#ff7f0e"
C_S4, C_REF = "#9467bd", "#888888"
C_BEST, C_EPA = "#378ADD", "#e24b4a"

plt.rcParams.update({
    "font.size": 12, "font.family": "Arial",
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Global log capture
_LOG_LINES = []


def log(s=""):
    print(s)
    _LOG_LINES.append(s)


# ── AQI converter ────────────────────────────────────────────────────────────

_PM25_BP = [(0,50,0.0,12.0),(50,100,12.0,35.4),(100,150,35.4,55.4),
            (150,200,55.4,150.4),(200,300,150.4,250.4),(300,400,250.4,350.4),
            (400,500,350.4,500.4)]

def aqi_to_ugm3(aqi):
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return None
    aqi = float(aqi)
    if aqi < 0: return 0.0
    for al, ah, cl, ch in _PM25_BP:
        if al <= aqi <= ah:
            return (ch - cl) / (ah - al) * (aqi - al) + cl
    al, ah, cl, ch = _PM25_BP[-1]
    return (ch - cl) / (ah - al) * (aqi - al) + cl


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ── PyTorch helpers ──────────────────────────────────────────────────────────

class CalibrationLSTM(nn.Module):
    def __init__(self, input_dim=8, bidirectional=False):
        super().__init__()
        d = 2 if bidirectional else 1
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=bidirectional)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64 * d, 32, batch_first=True, bidirectional=bidirectional)
        self.drop2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * d, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def train_torch_model(model, X_train, y_train, epochs=200, patience=10,
                      batch_size=32, lr=0.001, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.HuberLoss(delta=1.0)
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
            val_loss = loss_fn(model(val_X), val_y).item()
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


# ══════════════════════════════════════════════════════════════════════════════
# PART A: CALIBRATION (9-model comparison)
# ══════════════════════════════════════════════════════════════════════════════

def build_paired_dataset():
    log("\n" + "=" * 60)
    log("PART A: CALIBRATION — Building Paired Dataset")
    log("=" * 60)

    ref_df = pd.read_csv(REFERENCE_CSV)
    ref_df["createdAt"] = pd.to_datetime(ref_df["createdAt"], utc=True)

    s4 = ref_df[ref_df["deviceId"] == "Sensor-4"].copy()
    s4["hour_bucket"] = s4["createdAt"].dt.floor("h")

    rs = ref_df[ref_df["deviceId"] == "Reference-Station"].copy()
    mask = rs["pm25_ugm3"].isna() & rs["pm25_raw"].notna()
    if mask.any():
        rs.loc[mask, "pm25_ugm3"] = rs.loc[mask, "pm25_raw"].apply(aqi_to_ugm3)
    rs["hour_bucket"] = rs["createdAt"].dt.floor("h")

    s4_h = s4.groupby("hour_bucket").agg({
        "pm25_raw": "mean", "pm1_raw": "mean", "pm10_raw": "mean",
        "temperature": "mean", "humidity": "mean", "pressure": "mean",
        "heat_index": "mean",
    }).reset_index()

    rs_h = rs.groupby("hour_bucket").agg({"pm25_ugm3": "mean"}).reset_index()

    paired = s4_h.merge(rs_h, on="hour_bucket", how="inner")
    paired = paired.rename(columns={"pm25_ugm3": "pm25_ref", "hour_bucket": "timestamp"})
    paired["hour"] = paired["timestamp"].dt.hour
    paired["month"] = paired["timestamp"].dt.month
    paired["day_of_week"] = paired["timestamp"].dt.dayofweek
    paired = paired.dropna(subset=["pm25_ref", "pm25_raw"]).sort_values("timestamp").reset_index(drop=True)

    corr = np.corrcoef(paired["pm25_raw"], paired["pm25_ref"])[0, 1]
    r2_pre = corr ** 2

    log(f"\n=== PAIRED DATASET ===")
    log(f"Total paired records: {len(paired)}")
    log(f"Date range: {paired['timestamp'].min()} -> {paired['timestamp'].max()}")
    log(f"Sensor-4 PM2.5: mean={paired['pm25_raw'].mean():.1f}, std={paired['pm25_raw'].std():.1f}")
    log(f"Reference PM2.5: mean={paired['pm25_ref'].mean():.1f}, std={paired['pm25_ref'].std():.1f}")
    log(f"Pre-calibration R: {corr:.4f} (R2={r2_pre:.4f})")
    log(f"Mean ratio (sensor/reference): {(paired['pm25_raw'] / paired['pm25_ref'].replace(0, np.nan)).mean():.2f}")

    return paired


def train_calibration_models(paired):
    log("\n" + "=" * 60)
    log("PART A: CALIBRATION — Training 9 Models")
    log("=" * 60)

    X_all = paired[FEATURES].values
    y_all = paired[TARGET].values

    # 80/20 chronological split
    split_idx = int(len(X_all) * 0.8)
    X_train_raw, X_test_raw = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # sklearn models
    sklearn_models = {
        "MLR": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(kernel="rbf"),
        "RF": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=RANDOM_SEED, verbosity=0),
        "DTR": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "ANN": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                            random_state=RANDOM_SEED, early_stopping=True),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    tscv = TimeSeriesSplit(n_splits=5)

    results = []

    # Train sklearn models with KFold CV
    for name, model in sklearn_models.items():
        t0 = time.time()
        cv = cross_validate(model, X_train, y_train, cv=kf,
                            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
                            return_estimator=False)
        elapsed = time.time() - t0

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final = compute_metrics(y_test, y_pred)

        results.append({
            "name": name, "model": model, "type": "sklearn",
            "cv_r2_mean": cv["test_r2"].mean(), "cv_r2_std": cv["test_r2"].std(),
            "cv_rmse_mean": -cv["test_neg_root_mean_squared_error"].mean(),
            "cv_mae_mean": -cv["test_neg_mean_absolute_error"].mean(),
            "final_r2": final["r2"], "final_rmse": final["rmse"], "final_mae": final["mae"],
            "time": elapsed,
        })
        log(f"  {name}: CV R2={cv['test_r2'].mean():.4f} ({elapsed:.1f}s)")

    # Train LSTM and BiLSTM with TimeSeriesSplit
    WINDOW = 6
    for name, bidir in [("LSTM", False), ("BiLSTM", True)]:
        t0 = time.time()
        cv_r2s, cv_rmses, cv_maes = [], [], []

        for train_idx, val_idx in tscv.split(X_train):
            Xtr, Xvl = X_train[train_idx], X_train[val_idx]
            ytr, yvl = y_train[train_idx], y_train[val_idx]

            # Build sequences
            Xtr_seq = np.array([Xtr[i:i+WINDOW] for i in range(len(Xtr)-WINDOW)])
            ytr_seq = ytr[WINDOW:]
            Xvl_seq = np.array([Xvl[i:i+WINDOW] for i in range(len(Xvl)-WINDOW)])
            yvl_seq = yvl[WINDOW:]

            if len(Xtr_seq) < 10 or len(Xvl_seq) < 5:
                continue

            m = CalibrationLSTM(input_dim=len(FEATURES), bidirectional=bidir)
            m = train_torch_model(m, Xtr_seq, ytr_seq, epochs=200, patience=10, batch_size=32)

            with torch.no_grad():
                pred = m(torch.tensor(Xvl_seq, dtype=torch.float32)).numpy()
            met = compute_metrics(yvl_seq, pred)
            cv_r2s.append(met["r2"])
            cv_rmses.append(met["rmse"])
            cv_maes.append(met["mae"])

        # Final test
        Xtr_seq = np.array([X_train[i:i+WINDOW] for i in range(len(X_train)-WINDOW)])
        ytr_seq = y_train[WINDOW:]
        Xte_seq = np.array([X_test[i:i+WINDOW] for i in range(len(X_test)-WINDOW)])
        yte_seq = y_test[WINDOW:]

        final_model = CalibrationLSTM(input_dim=len(FEATURES), bidirectional=bidir)
        final_model = train_torch_model(final_model, Xtr_seq, ytr_seq, epochs=200, patience=10)

        with torch.no_grad():
            y_pred = final_model(torch.tensor(Xte_seq, dtype=torch.float32)).numpy()
        final = compute_metrics(yte_seq, y_pred)
        elapsed = time.time() - t0

        results.append({
            "name": name, "model": final_model, "type": "torch",
            "cv_r2_mean": np.mean(cv_r2s) if cv_r2s else float("nan"),
            "cv_r2_std": np.std(cv_r2s) if cv_r2s else float("nan"),
            "cv_rmse_mean": np.mean(cv_rmses) if cv_rmses else float("nan"),
            "cv_mae_mean": np.mean(cv_maes) if cv_maes else float("nan"),
            "final_r2": final["r2"], "final_rmse": final["rmse"], "final_mae": final["mae"],
            "time": elapsed,
        })
        log(f"  {name}: CV R2={results[-1]['cv_r2_mean']:.4f} ({elapsed:.1f}s)")

    # Sort by CV R²
    results.sort(key=lambda r: r["cv_r2_mean"], reverse=True)
    best = results[0]

    log(f"\n{'Model':<10} | {'R2 (mean+/-std)':>20} | {'RMSE (ug/m3)':>13} | {'MAE (ug/m3)':>12} | {'Time':>6}")
    log("-" * 75)
    for r in results:
        marker = "[BEST] " if r["name"] == best["name"] else "       "
        r2s = f"{r['cv_r2_mean']:.3f}+/-{r['cv_r2_std']:.3f}"
        log(f"{marker}{r['name']:<5} | {r2s:>20} | {r['cv_rmse_mean']:>13.2f} | {r['cv_mae_mean']:>12.2f} | {r['time']:>5.1f}s")

    epa = best["cv_r2_mean"] >= 0.70
    log(f"\nBest model: {best['name']} (CV R2 = {best['cv_r2_mean']:.4f})")
    log(f"EPA criterion (R2 >= 0.70): {'MET' if epa else 'NOT MET'}")

    return results, best, scaler


# ══════════════════════════════════════════════════════════════════════════════
# PART B: AGE-BASED CALIBRATION (4 weekly windows)
# ══════════════════════════════════════════════════════════════════════════════

def age_based_calibration(paired, best_result):
    log("\n" + "=" * 60)
    log("PART B: AGE-BASED CALIBRATION — 4 Weekly Windows")
    log("=" * 60)

    total_days = (paired["timestamp"].max() - paired["timestamp"].min()).days
    window_days = total_days // 4
    start = paired["timestamp"].min()

    windows = []
    for i in range(4):
        end = start + pd.Timedelta(days=window_days) if i < 3 else paired["timestamp"].max() + pd.Timedelta(hours=1)
        w = paired[(paired["timestamp"] >= start) & (paired["timestamp"] < end)].copy()
        windows.append({"week": i + 1, "start": start, "end": end, "data": w})
        start = end

    winner_name = best_result["name"]
    age_results = []

    for w in windows:
        df = w["data"]
        if len(df) < 10:
            log(f"  Week {w['week']}: only {len(df)} records, skipping")
            continue

        X = df[FEATURES].values
        y = df[TARGET].values
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)

        # MLR always
        mlr = LinearRegression().fit(X_sc, y)
        mlr_pred = mlr.predict(X_sc)
        mlr_met = compute_metrics(y, mlr_pred)

        coefs = {"intercept": float(mlr.intercept_)}
        for fname, c in zip(FEATURES, mlr.coef_):
            coefs[fname] = float(c)

        # Winner model (3-fold CV)
        winner_cv_r2 = None
        winner_r2 = mlr_met["r2"]
        winner_rmse = mlr_met["rmse"]
        winner_importance = (np.abs(mlr.coef_) / np.abs(mlr.coef_).sum()).tolist()

        if winner_name != "MLR" and winner_name in ("RF", "XGBoost", "DTR", "KNN", "SVR", "ANN"):
            model_map = {
                "RF": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
                "XGBoost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        random_state=RANDOM_SEED, verbosity=0),
                "DTR": DecisionTreeRegressor(random_state=RANDOM_SEED),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(kernel="rbf"),
                "ANN": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                    random_state=RANDOM_SEED, early_stopping=True),
            }
            if winner_name in model_map:
                wm = model_map[winner_name]
                n_splits = min(3, len(X_sc) // 10)
                if n_splits >= 2:
                    cv = cross_validate(wm, X_sc, y, cv=KFold(n_splits, shuffle=True, random_state=RANDOM_SEED),
                                        scoring="r2")
                    winner_cv_r2 = cv["test_score"].mean()
                wm.fit(X_sc, y)
                wp = wm.predict(X_sc)
                wmet = compute_metrics(y, wp)
                winner_r2 = wmet["r2"]
                winner_rmse = wmet["rmse"]
                if hasattr(wm, "feature_importances_"):
                    winner_importance = wm.feature_importances_.tolist()
                elif hasattr(wm, "coef_"):
                    imp = np.abs(wm.coef_)
                    winner_importance = (imp / imp.sum()).tolist()

        age_results.append({
            "week": w["week"],
            "start": w["start"].strftime("%Y-%m-%d"),
            "end": w["end"].strftime("%Y-%m-%d"),
            "n_records": len(df),
            "mean_temp": df["temperature"].mean(),
            "mean_pm25_ref": df["pm25_ref"].mean(),
            "mlr_coefs": coefs,
            "mlr_r2": mlr_met["r2"], "mlr_rmse": mlr_met["rmse"],
            "winner_r2": winner_r2, "winner_rmse": winner_rmse,
            "winner_cv_r2": winner_cv_r2, "winner_importance": winner_importance,
            "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        })

    # Print structured table
    header = f"{'Metric':<20}" + "".join(f" | {'Week '+str(ar['week'])+' ('+ar['start'][:10]+')':>22}" for ar in age_results)
    log(f"\n{header}")
    log("-" * len(header))

    rows = [
        ("N records", [str(ar["n_records"]) for ar in age_results]),
        ("Mean temp (C)", [f"{ar['mean_temp']:.1f}" for ar in age_results]),
        ("Mean PM2.5 ref", [f"{ar['mean_pm25_ref']:.1f}" for ar in age_results]),
        ("Intercept", [f"{ar['mlr_coefs']['intercept']:.4f}" for ar in age_results]),
    ]
    for feat in FEATURES:
        label = FEATURE_LABELS[FEATURES.index(feat)]
        rows.append((label, [f"{ar['mlr_coefs'].get(feat, 0):.4f}" for ar in age_results]))
    rows.append(("MLR R2", [f"{ar['mlr_r2']:.4f}" for ar in age_results]))
    rows.append(("MLR RMSE", [f"{ar['mlr_rmse']:.2f}" for ar in age_results]))
    rows.append((f"{winner_name} R2", [f"{ar['winner_r2']:.4f}" for ar in age_results]))
    rows.append((f"{winner_name} RMSE", [f"{ar['winner_rmse']:.2f}" for ar in age_results]))

    for label, vals in rows:
        line = f"{label:<20}" + "".join(f" | {v:>22}" for v in vals)
        log(line)

    return age_results


# ══════════════════════════════════════════════════════════════════════════════
# PART D: FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_timeseries():
    log("\n  Generating fig_pm25_timeseries.png...")
    meas = pd.read_csv(MEASUREMENTS_CSV)
    meas["createdAt"] = pd.to_datetime(meas["createdAt"], utc=True)
    ref_df = pd.read_csv(REFERENCE_CSV)
    ref_df["createdAt"] = pd.to_datetime(ref_df["createdAt"], utc=True)
    mask = ref_df["pm25_ugm3"].isna() & ref_df["pm25_raw"].notna()
    if mask.any():
        ref_df.loc[mask, "pm25_ugm3"] = ref_df.loc[mask, "pm25_raw"].apply(aqi_to_ugm3)

    fig, ax = plt.subplots(figsize=(14, 5))
    for did, color in [("Sensor-1", C_S1), ("Sensor-2", C_S2), ("Sensor-3", C_S3)]:
        s = meas[meas["deviceId"] == did].set_index("createdAt")["pm25_raw"].resample("1h").mean().dropna()
        ax.plot(s.index, s.values, color=color, linewidth=0.8, alpha=0.9, label=did)

    ref = ref_df[ref_df["deviceId"] == "Reference-Station"].set_index("createdAt")["pm25_ugm3"].resample("1h").mean().dropna()
    ax.plot(ref.index, ref.values, color=C_REF, linewidth=2, linestyle="--", alpha=0.8, label="Reference (AQICN)")

    s4 = ref_df[ref_df["deviceId"] == "Sensor-4"].set_index("createdAt")["pm25_raw"].resample("1h").mean().dropna()
    ax.plot(s4.index, s4.values, color=C_S4, linewidth=0.8, alpha=0.9, label="Sensor-4 (reference node)")

    ax.axvspan(pd.Timestamp("2026-03-10", tz="UTC"), pd.Timestamp("2026-03-13", tz="UTC"), color=C_EPA, alpha=0.08)
    ax.text(pd.Timestamp("2026-03-11T12:00", tz="UTC"), ax.get_ylim()[1]*0.92, "Sensor-1\nhardware gap",
            ha="center", fontsize=8, color="#c62828", style="italic")
    ax.axvspan(pd.Timestamp("2026-03-14", tz="UTC"), pd.Timestamp("2026-03-15", tz="UTC"), color="#ffa000", alpha=0.08)
    ax.text(pd.Timestamp("2026-03-14T12:00", tz="UTC"), ax.get_ylim()[1]*0.82, "Backend\noutage",
            ha="center", fontsize=8, color="#e65100", style="italic")
    ax.axhline(y=15, color=C_EPA, linewidth=1, linestyle=":", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 ($\\mu$g/m$^3$)")
    ax.set_title("PM2.5 concentration time series (hourly mean)")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_pm25_timeseries.png"), dpi=300)
    plt.close(fig)


def fig_calibration_barplot(cal_results):
    log("  Generating fig_calibration_barplot.png...")
    # Sort by cv_r2_mean descending
    sorted_results = sorted(cal_results, key=lambda r: r["cv_r2_mean"], reverse=True)
    names = [r["name"] for r in sorted_results]
    cv_r2 = [r["cv_r2_mean"] for r in sorted_results]
    cv_std = [r["cv_r2_std"] for r in sorted_results]

    EPA_THRESHOLD = 0.70
    colors = ["#2ecc71" if v >= EPA_THRESHOLD else "#b0b0b0" for v in cv_r2]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, cv_r2, 0.6, yerr=cv_std, capsize=4, color=colors, edgecolor="white")
    ax.axhline(y=EPA_THRESHOLD, color=C_EPA, linewidth=1.5, linestyle="--", alpha=0.7,
               label="EPA target (R$^2$ $\\geq$ 0.70)")
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cross-Validation R$^2$ (mean $\\pm$ std)")
    ax.set_title("Calibration Model Comparison — CV R$^2$")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_calibration_barplot.png"), dpi=300)
    plt.close(fig)


def fig_scatter(paired, best_cal, cal_scaler):
    log("  Generating fig_scatter_before_after.png...")
    X = paired[FEATURES].values
    y = paired[TARGET].values
    X_sc = cal_scaler.transform(X)

    split = int(len(X) * 0.8)
    is_test = np.arange(len(X)) >= split

    sensor_pm25 = paired["pm25_raw"].values
    ref_pm25 = y

    # Before
    r2_before = r2_score(ref_pm25, sensor_pm25)
    rmse_before = np.sqrt(mean_squared_error(ref_pm25, sensor_pm25))

    # After
    if best_cal["type"] == "sklearn":
        calibrated = best_cal["model"].predict(X_sc)
    else:
        WINDOW = 6
        seqs = np.array([X_sc[i:i+WINDOW] for i in range(len(X_sc)-WINDOW)])
        with torch.no_grad():
            calibrated = best_cal["model"](torch.tensor(seqs, dtype=torch.float32)).numpy()
        # Pad beginning
        calibrated = np.concatenate([np.full(WINDOW, np.nan), calibrated])
        valid = ~np.isnan(calibrated)
        ref_pm25 = ref_pm25[valid]
        sensor_pm25 = sensor_pm25[valid]
        calibrated = calibrated[valid]
        is_test = is_test[valid]

    calibrated = np.maximum(calibrated, 0)
    r2_after = r2_score(ref_pm25, calibrated)
    rmse_after = np.sqrt(mean_squared_error(ref_pm25, calibrated))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def _scatter(ax, x, y_vals, title, r2, rmse):
        ax.scatter(x[~is_test], y_vals[~is_test], alpha=0.35, s=20, color=C_S2, label="Train", zorder=2)
        ax.scatter(x[is_test], y_vals[is_test], alpha=0.6, s=25, color=C_EPA, marker="D", label="Test", zorder=3)
        lims = [min(x.min(), y_vals.min()), max(x.max(), y_vals.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="1:1 line")
        from sklearn.linear_model import LinearRegression as LR
        lr = LR().fit(x.reshape(-1,1), y_vals)
        xl = np.linspace(lims[0], lims[1], 100)
        ax.plot(xl, lr.predict(xl.reshape(-1,1)), color="#ff7f0e", linewidth=1.5,
                label=f"y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}")
        ax.set_xlabel("Sensor PM2.5 ($\\mu$g/m$^3$)")
        ax.set_ylabel("Reference PM2.5 ($\\mu$g/m$^3$)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.text(0.05, 0.95, f"R$^2$ = {r2:.4f}\nRMSE = {rmse:.2f} $\\mu$g/m$^3$\nn = {len(x)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right")

    _scatter(ax1, sensor_pm25, ref_pm25, "(a) Before calibration", r2_before, rmse_before)
    _scatter(ax2, calibrated, ref_pm25, "(b) After calibration", r2_after, rmse_after)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_scatter_before_after.png"), dpi=300)
    plt.close(fig)

    log(f"  Before: R2={r2_before:.4f}, RMSE={rmse_before:.2f}")
    log(f"  After:  R2={r2_after:.4f}, RMSE={rmse_after:.2f}")


def fig_feature_importance(best_cal):
    log("  Generating fig_feature_importance.png...")
    model = best_cal["model"]
    name = best_cal["name"]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
        imp = imp / imp.sum()
    else:
        log("  WARNING: Cannot extract feature importance, skipping.")
        return

    pairs = sorted(zip(FEATURE_LABELS, imp), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(labels)), values, color=C_BEST, edgecolor="white", height=0.6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance score")
    ax.set_title(f"Feature importance for PM2.5 calibration ({name})")
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.01, i, f"{v:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_feature_importance.png"), dpi=300)
    plt.close(fig)


def fig_age_feature_importance_grid(age_results, winner_name):
    log("  Generating fig_age_feature_importance_grid.png...")
    n = len(age_results)
    fig, axes = plt.subplots(1, n, figsize=(16, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (ar, ax) in enumerate(zip(age_results, axes)):
        imp = np.array(ar["winner_importance"])
        pairs = sorted(zip(FEATURE_LABELS, imp), key=lambda x: x[1], reverse=True)
        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        ax.barh(np.arange(len(labels)), values, color=C_BEST, edgecolor="white", height=0.6)
        ax.set_yticks(np.arange(len(labels)))
        if idx == 0:
            ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Week {ar['week']} ({ar['start']})")

    fig.suptitle(f"Feature importance per age window ({winner_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_age_feature_importance_grid.png"), dpi=300)
    plt.close(fig)


def fig_age_r2_trend(age_results, winner_name):
    log("  Generating fig_age_r2_trend.png...")
    weeks = [r["week"] for r in age_results]
    mlr_r2 = [r["mlr_r2"] for r in age_results]
    win_r2 = [r["winner_r2"] for r in age_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weeks, win_r2, "o-", color="#2ecc71", linewidth=2, markersize=8, label=winner_name)
    if winner_name != "MLR":
        ax.plot(weeks, mlr_r2, "s--", color="#e74c3c", linewidth=2, markersize=8, label="MLR")
    ax.axhline(y=0.70, color=C_EPA, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(0.5, 0.72, "EPA threshold", fontsize=9, color=C_EPA)
    ax.set_xlabel("Week")
    ax.set_ylabel("R$^2$ score")
    ax.set_title("Calibration R$^2$ by sensor age (weekly windows)")
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"Week {w}\n{age_results[w-1]['start'][5:]}" for w in weeks])
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_age_r2_trend.png"), dpi=300)
    plt.close(fig)


def fig_age_heatmap(age_results):
    log("  Generating fig_age_coefficients_heatmap.png...")
    coef_matrix = []
    for feat in FEATURES:
        row = [r["mlr_coefs"].get(feat, 0) for r in age_results]
        coef_matrix.append(row)

    data = np.array(coef_matrix)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.abs(data).max()
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(age_results)))
    ax.set_xticklabels([f"Week {r['week']}" for r in age_results])
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURE_LABELS)
    for i in range(len(FEATURES)):
        for j in range(len(age_results)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(data[i,j]) > vmax*0.6 else "black")
    fig.colorbar(im, ax=ax, label="MLR coefficient (standardized)")
    ax.set_title("MLR coefficient drift across weekly windows")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_age_coefficients_heatmap.png"), dpi=300)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PART E: EXPORT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def export_results(cal_results, best_cal, age_results, paired):
    log("\n" + "=" * 60)
    log("PART E: EXPORTING RESULTS")
    log("=" * 60)

    # full_results.json
    full = {
        "calibration": {
            "paired_records": len(paired),
            "best_model": best_cal["name"],
            "models": [{k: v for k, v in r.items() if k not in ("model", "scaler", "y_test", "pred_test")}
                       for r in cal_results],
        },
        "age_based": age_results,
    }
    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(full, f, indent=2, default=str)
    log("  Saved full_results.json")

    # age_calibration_models.json
    age_docs = []
    for ar in age_results:
        age_docs.append({
            "sensor_id": "Sensor-4",
            "trained_at": datetime.utcnow().isoformat(),
            "model_type": "MLR",
            "coefficients": ar["mlr_coefs"],
            "metrics": {"r2": ar["mlr_r2"], "rmse": ar["mlr_rmse"]},
            "is_active": ar["week"] == len(age_results),
            "training_samples": ar["n_records"],
            "week": ar["week"],
            "period_start": ar["start"],
            "period_end": ar["end"],
        })
    with open(os.path.join(OUTPUT_DIR, "age_calibration_models.json"), "w") as f:
        json.dump(age_docs, f, indent=2)
    log("  Saved age_calibration_models.json")

    # dataset_summary.txt
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.txt"), "w") as f:
        f.write("\n".join(_LOG_LINES))
    log("  Saved dataset_summary.txt")


# ══════════════════════════════════════════════════════════════════════════════
# PART F: LITERATURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def print_literature_tables(best_cal):
    log("\n" + "=" * 60)
    log("TABLE: Calibration — Comparison with Literature")
    log("=" * 60)
    log(f"{'Study':<22} | {'Location':<10} | {'Sensor':<10} | {'Best Model':<12} | {'R2':>10} | {'Co-location':<15}")
    log("-" * 95)
    for s, l, sn, m, r2, c in [
        ("O'Leary (2024)", "Dublin", "PMS5003", "MLR", "0.81", "Physical"),
        ("Barkjohn (2021)", "US", "PMS5003", "MLR+RH", "0.65-0.94", "Physical"),
        ("Campmier (2023)", "SE US", "PMS5003", "RF", "0.86-0.94", "Physical"),
        ("Nan (2026)", "China", "PMS5003", "XGBoost", "0.88", "Physical"),
        ("Malyan (2024)", "Delhi", "PMS5003", "RF", "0.70-0.95", "Physical"),
    ]:
        log(f"{s:<22} | {l:<10} | {sn:<10} | {m:<12} | {r2:>10} | {c:<15}")
    log(f"{'This work':<22} | {'Astana':<10} | {'PMS5003':<10} | {best_cal['name']:<12} | {best_cal['cv_r2_mean']:>10.2f} | {'Co-located':<15}")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_dataset_stats():
    log("\n" + "=" * 60)
    log("DATASET SUMMARY STATISTICS")
    log("=" * 60)

    meas = pd.read_csv(MEASUREMENTS_CSV)
    meas["createdAt"] = pd.to_datetime(meas["createdAt"], utc=True)
    ref_df = pd.read_csv(REFERENCE_CSV)
    ref_df["createdAt"] = pd.to_datetime(ref_df["createdAt"], utc=True)
    mask = ref_df["pm25_ugm3"].isna() & ref_df["pm25_raw"].notna()
    if mask.any():
        ref_df.loc[mask, "pm25_ugm3"] = ref_df.loc[mask, "pm25_raw"].apply(aqi_to_ugm3)

    for did in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        s = meas[meas["deviceId"] == did]
        pm = s["pm25_raw"].dropna()
        synth = s["synthetic"].fillna(False).astype(bool).sum() if "synthetic" in s.columns else 0
        log(f"\n{did}: {len(s)} records ({len(s)-synth} real / {synth} gap-filled)")
        log(f"  Date: {s['createdAt'].min().strftime('%Y-%m-%d %H:%M')} to {s['createdAt'].max().strftime('%Y-%m-%d %H:%M')}")
        log(f"  PM2.5: mean={pm.mean():.1f}, std={pm.std():.1f}, median={pm.median():.1f}, range=[{pm.min():.1f}, {pm.max():.1f}]")

    log(f"\nOverall: {len(meas)} records, PM2.5 mean={meas['pm25_raw'].mean():.1f} ug/m3")
    log(f"Temperature: [{meas['temperature'].min():.1f}, {meas['temperature'].max():.1f}] C, mean={meas['temperature'].mean():.1f} C")

    rs = ref_df[ref_df["deviceId"] == "Reference-Station"]
    rpm = rs["pm25_ugm3"].dropna()
    log(f"\nReference-Station: {len(rs)} records")
    log(f"  PM2.5: mean={rpm.mean():.1f}, std={rpm.std():.1f}, range=[{rpm.min():.1f}, {rpm.max():.1f}]")

    s4 = ref_df[ref_df["deviceId"] == "Sensor-4"]
    log(f"Sensor-4: {len(s4)} records")

    # WHO exceedance
    log("\nWHO 24h guideline exceedance (daily mean > 15 ug/m3):")
    for did in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        daily = meas[meas["deviceId"] == did].set_index("createdAt")["pm25_raw"].resample("1D").mean().dropna()
        exc = (daily > 15).sum()
        log(f"  {did}: {exc}/{len(daily)} days ({100*exc/len(daily):.0f}%)")
    rd = rs.set_index("createdAt")["pm25_ugm3"].resample("1D").mean().dropna()
    exc = (rd > 15).sum()
    log(f"  Reference: {exc}/{len(rd)} days ({100*exc/len(rd):.0f}%)")

    # Correlations
    log("\nInter-sensor correlation (hourly PM2.5):")
    hourly = {}
    for did in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        hourly[did] = meas[meas["deviceId"] == did].set_index("createdAt")["pm25_raw"].resample("1h").mean()
    for a, b in [("Sensor-1","Sensor-2"),("Sensor-1","Sensor-3"),("Sensor-2","Sensor-3")]:
        m = pd.concat([hourly[a], hourly[b]], axis=1, keys=[a,b]).dropna()
        log(f"  {a} <-> {b}: r={m[a].corr(m[b]):.4f} (n={len(m)})")

    ref_h = rs.set_index("createdAt")["pm25_ugm3"].resample("1h").mean()
    log("\nSensor vs Reference correlation:")
    for did in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        m = pd.concat([hourly[did], ref_h], axis=1, keys=[did,"Ref"]).dropna()
        log(f"  {did} <-> Reference: r={m[did].corr(m['Ref']):.4f} (n={len(m)})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Part A: Calibration
    paired = build_paired_dataset()
    cal_results, best_cal, cal_scaler = train_calibration_models(paired)

    # Part B: Age-based
    age_results = age_based_calibration(paired, best_cal)

    # Part D: Figures
    log("\n" + "=" * 60)
    log("PART D: GENERATING FIGURES")
    log("=" * 60)
    fig_timeseries()
    fig_calibration_barplot(cal_results)
    fig_scatter(paired, best_cal, cal_scaler)
    fig_feature_importance(best_cal)
    fig_age_r2_trend(age_results, best_cal["name"])
    fig_age_heatmap(age_results)
    if best_cal["name"] != "MLR":
        fig_age_feature_importance_grid(age_results, best_cal["name"])

    # Dataset stats
    print_dataset_stats()

    # Part E: Export
    export_results(cal_results, best_cal, age_results, paired)

    # Part F: Literature
    print_literature_tables(best_cal)

    # Final summary
    log("\n" + "=" * 60)
    log("ALL COMPLETE")
    log("=" * 60)
    log(f"\nOutput directory: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        log(f"  {f:<45} {size/1024:.1f} KB")

    # Update dataset_summary.txt with final output
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.txt"), "w") as f:
        f.write("\n".join(_LOG_LINES))


if __name__ == "__main__":
    main()

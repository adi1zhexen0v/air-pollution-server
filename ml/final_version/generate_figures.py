"""Generate all thesis Chapter 4 figures and dataset statistics."""

import json
import math
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# ── Config ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MEASUREMENTS_CSV = os.path.join(BASE_DIR, "final.measurements.csv")
REFERENCE_CSV = os.path.join(BASE_DIR, "final.referencemeasurements.csv")
CALIBRATION_RESULTS_JSON = os.path.join(BASE_DIR, "..", "virtual_colocation", "output", "calibration_results.json")
VIRTUAL_DATA_JSON = os.path.join(BASE_DIR, "..", "virtual_colocation", "output", "virtual_colocation_data.json")
MODEL_PKL = os.path.join(BASE_DIR, "..", "virtual_colocation", "output", "best_calibration_model.pkl")
SCALER_PKL = os.path.join(BASE_DIR, "..", "virtual_colocation", "output", "feature_scaler.pkl")
FORECASTING_HORIZON_CSV = os.path.join(BASE_DIR, "..", "selection", "outputs", "forecasting_per_horizon.csv")
FORECASTING_COMPARISON_CSV = os.path.join(BASE_DIR, "..", "selection", "outputs", "results", "forecasting_comparison.csv")

# Colors
C_SENSOR1 = "#2ca02c"
C_SENSOR2 = "#1f77b4"
C_SENSOR3 = "#ff7f0e"
C_REFERENCE = "#888888"
C_CALIBRATED = "#e24b4a"
C_EPA = "#e24b4a"

# ── AQI → µg/m³ converter (EPA PM2.5 breakpoints) ───────────────────────────

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


# ── Data loading helpers ─────────────────────────────────────────────────────


def load_measurements():
    df = pd.read_csv(MEASUREMENTS_CSV)
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    return df


def load_reference():
    df = pd.read_csv(REFERENCE_CSV)
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    # Ensure pm25_ugm3 is filled
    mask = df["pm25_ugm3"].isna() & df["pm25_raw"].notna()
    if mask.any():
        df.loc[mask, "pm25_ugm3"] = df.loc[mask, "pm25_raw"].apply(aqi_to_ugm3_pm25)
    return df


# ── Figure 4.1: PM2.5 Time Series ───────────────────────────────────────────


def figure_4_1_timeseries(meas_df, ref_df):
    print("\n=== Figure 4.1: PM2.5 Time Series ===")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot order: Sensor-1..4, Reference, WHO
    sensor_colors = {
        "Sensor-1": C_SENSOR1,
        "Sensor-2": C_SENSOR2,
        "Sensor-3": C_SENSOR3,
    }

    for device_id, color in sensor_colors.items():
        s = meas_df[meas_df["deviceId"] == device_id].copy()
        s = s.set_index("createdAt")["pm25_raw"].resample("1h").mean().dropna()
        ax.plot(s.index, s.values, color=color, linewidth=0.8, alpha=0.9, label=device_id)

    s4 = ref_df[ref_df["deviceId"] == "Sensor-4"].copy()
    s4 = s4.set_index("createdAt")["pm25_raw"].resample("1h").mean().dropna()
    ax.plot(s4.index, s4.values, color="#9467bd", linewidth=0.8, alpha=0.9,
            label="Sensor-4 (reference node)")

    ref = ref_df[ref_df["deviceId"] == "Reference-Station"].copy()
    ref = ref.set_index("createdAt")["pm25_ugm3"].resample("1h").mean().dropna()
    ax.plot(ref.index, ref.values, color=C_REFERENCE, linewidth=2, linestyle="--",
            alpha=0.8, label="Reference (AQICN)")

    # WHO guideline — in legend, no separate text label
    ax.axhline(y=15, color=C_EPA, linewidth=1.5, linestyle="--", alpha=0.6,
               label="WHO 24h guideline (15 $\mu$g/m$^3$)")

    # Shaded event regions (exact dates from fill-data-gaps.js SKIP_RANGES)
    ax.axvspan(pd.Timestamp("2026-03-10T02:09:23", tz="UTC"),
               pd.Timestamp("2026-03-13T04:23:10", tz="UTC"),
               color="#e24b4a", alpha=0.08, label="_nolegend_")
    ax.text(pd.Timestamp("2026-03-11T15:00", tz="UTC"), ax.get_ylim()[1] * 0.92,
            "Sensor-1\nhardware gap", ha="center", fontsize=8, color="#c62828", style="italic")

    ax.axvspan(pd.Timestamp("2026-03-14T04:00:00", tz="UTC"),
               pd.Timestamp("2026-03-15T11:00:00", tz="UTC"),
               color="#ffa000", alpha=0.08, label="_nolegend_")
    ax.text(pd.Timestamp("2026-03-14T19:30", tz="UTC"), ax.get_ylim()[1] * 0.82,
            "Backend\noutage", ha="center", fontsize=8, color="#e65100", style="italic")

    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 ($\mu$g/m$^3$)")
    ax.set_title("PM2.5 concentration time series (hourly mean)")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    # Remove x-axis padding so data fills the full width
    all_times = []
    for device_id in sensor_colors:
        s = meas_df[meas_df["deviceId"] == device_id]
        all_times.extend(s["createdAt"].tolist())
    all_times.extend(ref.index.tolist())
    if all_times:
        ax.set_xlim(min(all_times), max(all_times))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_4_1_pm25_timeseries.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4.3a: Calibration Model Comparison ───────────────────────────────


def figure_4_3_calibration_comparison():
    print("\n=== Figure 4.3: Calibration Model Comparison ===")

    with open(CALIBRATION_RESULTS_JSON) as f:
        data = json.load(f)

    models = sorted(data["models"], key=lambda m: m["cv_r2_mean"], reverse=True)
    names = [m["name"] for m in models]
    cv_r2 = [m["cv_r2_mean"] for m in models]
    cv_r2_std = [m["cv_r2_std"] for m in models]
    final_r2 = [m["final_r2"] for m in models]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, cv_r2, width, yerr=cv_r2_std, capsize=4,
                   color="#4c8cc4", edgecolor="white", label="CV R$^2$ (mean $\\pm$ std)")
    bars2 = ax.bar(x + width / 2, final_r2, width,
                   color="#f0a04b", edgecolor="white", label="Final Test R$^2$")

    ax.axhline(y=0.70, color=C_EPA, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(len(names) - 0.5, 0.72, "EPA threshold (R$^2$ $\\geq$ 0.70)",
            fontsize=9, color=C_EPA, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("R$^2$ score")
    ax.set_title("Calibration model comparison (co-location data)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_4_3_calibration_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4.3b: Scatter Before/After Calibration ───────────────────────────


def figure_4_3_scatter_calibration(ref_df):
    print("\n=== Figure 4.3b: Scatter Before/After Calibration ===")

    # Load Sensor-4 data (virtual colocation records)
    with open(VIRTUAL_DATA_JSON) as f:
        vdata = json.load(f)

    s4_records = []
    for r in vdata["data"]:
        ts = pd.to_datetime(r["createdAt"], utc=True)
        s4_records.append({
            "hour": ts.floor("h"),
            "sensor_pm25": r["pm25_raw"],
            "humidity": r.get("humidity"),
            "temperature": r.get("temperature"),
            "pressure": r.get("pressure"),
            "heat_index": r.get("heat_index"),
        })
    s4_df = pd.DataFrame(s4_records)
    s4_df["ts_hour"] = s4_df["hour"].dt.tz_localize(None)

    # Load reference station hourly
    ref_st = ref_df[ref_df["deviceId"] == "Reference-Station"].copy()
    ref_st["ts_hour"] = ref_st["createdAt"].dt.floor("h").dt.tz_localize(None)
    ref_hourly = ref_st.groupby("ts_hour")["pm25_ugm3"].mean().reset_index()
    ref_hourly.columns = ["ts_hour", "ref_pm25"]

    # Merge on hour
    paired = s4_df.merge(ref_hourly, on="ts_hour", how="inner")
    paired = paired.dropna(subset=["sensor_pm25", "ref_pm25"])
    print(f"Paired records: {len(paired)}")

    if len(paired) < 10:
        print("WARNING: Too few paired records for scatter plot, skipping.")
        return

    sensor_pm25 = paired["sensor_pm25"].values
    ref_pm25 = paired["ref_pm25"].values

    # 80/20 chronological split
    split_idx = int(len(paired) * 0.8)
    is_test = np.arange(len(paired)) >= split_idx

    # Before calibration metrics
    r2_before = r2_score(ref_pm25, sensor_pm25)
    rmse_before = np.sqrt(mean_squared_error(ref_pm25, sensor_pm25))

    # After calibration
    calibrated_pm25 = None
    r2_after = None
    rmse_after = None

    try:
        model = joblib.load(MODEL_PKL)
        scaler = joblib.load(SCALER_PKL)

        # Build feature matrix matching virtual_colocation.py FEATURE_COLS order
        paired["ts_dt"] = pd.to_datetime(paired["ts_hour"])
        features = pd.DataFrame({
            "virtual_pm25": paired["sensor_pm25"],
            "virtual_humidity": paired["humidity"],
            "virtual_temperature": paired["temperature"],
            "virtual_pressure": paired["pressure"],
            "virtual_heat_index": paired["heat_index"],
            "hour": paired["ts_dt"].dt.hour,
            "month": paired["ts_dt"].dt.month,
            "day_of_week": paired["ts_dt"].dt.dayofweek,
        }).fillna(0)

        X_scaled = scaler.transform(features.values)
        calibrated_pm25 = model.predict(X_scaled)
        calibrated_pm25 = np.maximum(calibrated_pm25, 0)

        r2_after = r2_score(ref_pm25, calibrated_pm25)
        rmse_after = np.sqrt(mean_squared_error(ref_pm25, calibrated_pm25))
        print(f"Before: R2={r2_before:.4f}, RMSE={rmse_before:.2f}")
        print(f"After:  R2={r2_after:.4f}, RMSE={rmse_after:.2f}")
    except Exception as e:
        print(f"WARNING: Could not load model/scaler: {e}")
        print("Generating single-panel plot only.")

    n_panels = 2 if calibrated_pm25 is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    def scatter_panel(ax, x, y, title, r2, rmse):
        # Train/test split coloring
        ax.scatter(x[~is_test], y[~is_test], alpha=0.35, s=20, color="#1f77b4", label="Train", zorder=2)
        ax.scatter(x[is_test], y[is_test], alpha=0.6, s=25, color="#e24b4a", marker="D", label="Test", zorder=3)

        # 1:1 line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="1:1 line")

        # Regression line
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_line, lr.predict(x_line.reshape(-1, 1)), color="#ff7f0e", linewidth=1.5,
                label=f"y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}")

        ax.set_xlabel("Sensor PM2.5 ($\mu$g/m$^3$)")
        ax.set_ylabel("Reference PM2.5 ($\mu$g/m$^3$)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Annotations
        ax.text(0.05, 0.95, f"R$^2$ = {r2:.4f}\nRMSE = {rmse:.2f} $\mu$g/m$^3$\nn = {len(x)}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right")

    scatter_panel(axes[0], sensor_pm25, ref_pm25,
                  "(a) Before calibration", r2_before, rmse_before)

    if calibrated_pm25 is not None:
        scatter_panel(axes[1], calibrated_pm25, ref_pm25,
                      "(b) After calibration", r2_after, rmse_after)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_4_3_scatter_calibration.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4.4: Forecasting Per-Horizon RMSE ────────────────────────────────


def figure_4_4_forecasting():
    print("\n=== Figure 4.4: Forecasting Per-Horizon RMSE ===")

    # Try per-horizon CSV first
    if os.path.exists(FORECASTING_HORIZON_CSV):
        df = pd.read_csv(FORECASTING_HORIZON_CSV)
        print(f"Loaded {FORECASTING_HORIZON_CSV}: {len(df)} rows")
    elif os.path.exists(FORECASTING_COMPARISON_CSV):
        # Fallback: extract per-day RMSE from comparison CSV
        comp = pd.read_csv(FORECASTING_COMPARISON_CSV)
        rows = []
        for _, row in comp.iterrows():
            for d in range(1, 8):
                col = f"rmse_day{d}"
                if col in row:
                    rows.append({"Model": row["model"], "Horizon_Day": d, "RMSE": row[col]})
        df = pd.DataFrame(rows)
        print(f"Extracted per-horizon data from comparison CSV: {len(df)} rows")
    else:
        print("ERROR: No forecasting data found, skipping figure.")
        return

    model_styles = {
        "XGBoost": {"color": "#2ca02c", "marker": "s"},
        "LSTM": {"color": "#1f77b4", "marker": "o"},
        "BiLSTM": {"color": "#9467bd", "marker": "^"},
        "CNN-LSTM": {"color": "#ff7f0e", "marker": "D"},
        "MLR": {"color": "#888888", "marker": "v"},
        "RF": {"color": "#d62728", "marker": "p"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, group in df.groupby("Model"):
        group = group.sort_values("Horizon_Day")
        style = model_styles.get(model_name, {"color": "black", "marker": "x"})
        ax.plot(group["Horizon_Day"], group["RMSE"],
                color=style["color"], marker=style["marker"],
                linewidth=2, markersize=7, label=model_name)

    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("RMSE ($\mu$g/m$^3$)")
    ax.set_title("Forecasting RMSE by horizon day")
    ax.set_xticks(range(1, 8))
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_4_4_forecasting_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Dataset Statistics ───────────────────────────────────────────────────────


def print_dataset_statistics(meas_df, ref_df):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # Per-sensor stats
    log("\n--- Per-Sensor Summary ---")
    for device_id in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        s = meas_df[meas_df["deviceId"] == device_id]
        pm = s["pm25_raw"].dropna()
        date_min = s["createdAt"].min()
        date_max = s["createdAt"].max()
        synth_count = s["synthetic"].fillna(False).astype(bool).sum() if "synthetic" in s.columns else 0
        real_count = len(s) - synth_count

        log(f"\n{device_id}:")
        log(f"  Total records:     {len(s)}")
        log(f"  Real / Synthetic:  {real_count} / {synth_count}")
        log(f"  Date range:        {date_min.strftime('%Y-%m-%d %H:%M')} to {date_max.strftime('%Y-%m-%d %H:%M')}")
        log(f"  PM2.5 mean (sd):   {pm.mean():.2f} ({pm.std():.2f}) ug/m3")
        log(f"  PM2.5 median:      {pm.median():.2f} ug/m3")
        log(f"  PM2.5 min / max:   {pm.min():.2f} / {pm.max():.2f} ug/m3")

    # Overall sensor stats
    log("\n--- Overall Sensor Statistics ---")
    all_pm = meas_df["pm25_raw"].dropna()
    log(f"Total sensor records: {len(meas_df)}")
    log(f"PM2.5 mean (sd):     {all_pm.mean():.2f} ({all_pm.std():.2f}) ug/m3")
    log(f"PM2.5 median:        {all_pm.median():.2f} ug/m3")
    log(f"PM2.5 min / max:     {all_pm.min():.2f} / {all_pm.max():.2f} ug/m3")

    # Temperature stats
    all_temp = meas_df["temperature"].dropna()
    log(f"\nTemperature min / max / mean: {all_temp.min():.1f} / {all_temp.max():.1f} / {all_temp.mean():.1f} C")

    # Reference station stats
    log("\n--- Reference Station ---")
    ref_st = ref_df[ref_df["deviceId"] == "Reference-Station"]
    ref_pm = ref_st["pm25_ugm3"].dropna()
    log(f"Total records:       {len(ref_st)}")
    log(f"Date range:          {ref_st['createdAt'].min().strftime('%Y-%m-%d')} to {ref_st['createdAt'].max().strftime('%Y-%m-%d')}")
    log(f"PM2.5 mean (sd):     {ref_pm.mean():.2f} ({ref_pm.std():.2f}) ug/m3")
    log(f"PM2.5 median:        {ref_pm.median():.2f} ug/m3")
    log(f"PM2.5 min / max:     {ref_pm.min():.2f} / {ref_pm.max():.2f} ug/m3")

    # Sensor-4 stats
    s4 = ref_df[ref_df["deviceId"] == "Sensor-4"]
    log(f"\n--- Co-located Sensor (Sensor-4) ---")
    log(f"Total records:       {len(s4)}")
    if len(s4) > 0:
        log(f"Date range:          {s4['createdAt'].min().strftime('%Y-%m-%d')} to {s4['createdAt'].max().strftime('%Y-%m-%d')}")

    # WHO exceedance (daily means > 15 ug/m3)
    log("\n--- WHO Exceedance Analysis ---")
    for device_id in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        s = meas_df[meas_df["deviceId"] == device_id].copy()
        s = s.set_index("createdAt")["pm25_raw"].resample("1D").mean().dropna()
        exceed_days = (s > 15).sum()
        total_days = len(s)
        log(f"{device_id}: {exceed_days}/{total_days} days exceeded WHO 24h guideline (15 ug/m3) = {100*exceed_days/total_days:.0f}%")

    ref_daily = ref_st.set_index("createdAt")["pm25_ugm3"].resample("1D").mean().dropna()
    exceed_ref = (ref_daily > 15).sum()
    log(f"Reference:  {exceed_ref}/{len(ref_daily)} days exceeded WHO 24h guideline = {100*exceed_ref/len(ref_daily):.0f}%")

    # Inter-sensor correlation
    log("\n--- Inter-Sensor Correlation (hourly PM2.5) ---")
    hourly = {}
    for device_id in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        s = meas_df[meas_df["deviceId"] == device_id].copy()
        s = s.set_index("createdAt")["pm25_raw"].resample("1h").mean()
        hourly[device_id] = s

    pairs = [("Sensor-1", "Sensor-2"), ("Sensor-1", "Sensor-3"), ("Sensor-2", "Sensor-3")]
    for a, b in pairs:
        merged = pd.concat([hourly[a], hourly[b]], axis=1, keys=[a, b]).dropna()
        if len(merged) > 0:
            corr = merged[a].corr(merged[b])
            log(f"  {a} <-> {b}: r = {corr:.4f} (n={len(merged)} hours)")

    # Sensor vs Reference correlation
    log("\n--- Sensor vs Reference Correlation (hourly PM2.5) ---")
    ref_hourly = ref_st.set_index("createdAt")["pm25_ugm3"].resample("1h").mean()
    for device_id in ["Sensor-1", "Sensor-2", "Sensor-3"]:
        merged = pd.concat([hourly[device_id], ref_hourly], axis=1, keys=[device_id, "Ref"]).dropna()
        if len(merged) > 0:
            corr = merged[device_id].corr(merged["Ref"])
            log(f"  {device_id} <-> Reference: r = {corr:.4f} (n={len(merged)} hours)")

    # Save to file
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {summary_path}")


# ── Calibration Results Table ────────────────────────────────────────────────


def print_calibration_table():
    print("\n" + "=" * 60)
    print("TABLE 4.3: Calibration Model Comparison (Co-location Data)")
    print("=" * 60)

    with open(CALIBRATION_RESULTS_JSON) as f:
        data = json.load(f)

    models = sorted(data["models"], key=lambda m: m["cv_r2_mean"], reverse=True)
    best = data["best_model"]

    header = f"{'Model':<12} | {'CV R2 (mean+/-std)':>20} | {'CV RMSE':>10} | {'CV MAE':>10} | {'Final R2':>10} | {'Final RMSE':>10} | {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    for m in models:
        marker = "[BEST] " if m["name"] == best else "       "
        r2_str = f"{m['cv_r2_mean']:.3f}+/-{m['cv_r2_std']:.3f}"
        print(f"{marker}{m['name']:<5} | {r2_str:>20} | {m['cv_rmse_mean']:>10.2f} | {m['cv_mae_mean']:>10.2f} | "
              f"{m['final_r2']:>10.4f} | {m['final_rmse']:>10.2f} | {m['training_time_sec']:>10.2f}")

    epa_pass = any(m["cv_r2_mean"] >= 0.70 for m in models)
    print(f"\nBest model: {best} (CV R2 = {next(m['cv_r2_mean'] for m in models if m['name'] == best):.4f})")
    print(f"EPA criterion met: {'Yes' if epa_pass else 'No'} (R2 >= 0.70)")
    print(f"Dataset: {data['dataset_size']} paired hourly records, {data['dataset_info']['train_size']} train / {data['dataset_info']['test_size']} test")

    print("\n" + "=" * 60)
    print("TABLE 4.X: Comparison with State-of-the-Art (skeleton)")
    print("=" * 60)

    print(f"{'Study':<22} | {'Location':<10} | {'Sensor':<10} | {'Best Model':<12} | {'R2':>10} | {'Co-location':<15}")
    print("-" * 95)
    lit = [
        ("O'Leary (2024)", "Dublin", "PMS5003", "MLR", "0.81", "Physical"),
        ("Barkjohn (2021)", "US", "PMS5003", "MLR+RH", "0.65-0.94", "Physical"),
        ("Campmier (2023)", "SE US", "PMS5003", "RF", "0.86-0.94", "Physical"),
        ("Nan (2026)", "China", "PMS5003", "XGBoost", "0.88", "Physical"),
        ("Malyan (2024)", "Delhi", "PMS5003", "RF", "0.70-0.95", "Physical"),
    ]
    for study, loc, sensor, model, r2, coloc in lit:
        print(f"{study:<22} | {loc:<10} | {sensor:<10} | {model:<12} | {r2:>10} | {coloc:<15}")

    best_r2 = next(m["cv_r2_mean"] for m in models if m["name"] == best)
    best_rmse = next(m["cv_rmse_mean"] for m in models if m["name"] == best)
    print(f"{'This work':<22} | {'Astana':<10} | {'PMS5003':<10} | {best:<12} | {best_r2:>10.2f} | {'Co-located':<15}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    meas_df = load_measurements()
    ref_df = load_reference()
    print(f"Measurements: {len(meas_df)} rows")
    print(f"Reference:    {len(ref_df)} rows")

    figure_4_1_timeseries(meas_df, ref_df)
    figure_4_3_calibration_comparison()
    figure_4_3_scatter_calibration(ref_df)
    figure_4_4_forecasting()
    print_dataset_statistics(meas_df, ref_df)
    print_calibration_table()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:<45} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()

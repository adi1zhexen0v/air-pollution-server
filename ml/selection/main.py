import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Ensure CWD is the ml/ directory regardless of where the script is invoked from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.extract.station_loader import collect_all_stations_data
from src.weather.open_meteo_fetcher import add_weather_columns
from src.preprocess.filter_data import filter_dataframe
from src.preprocess.time_features import extract_time_features
from src.preprocess.feature_selector import select_features
from src.preprocess.scaler import normalize_features
from src.preprocess.synthetic_sensor import generate_synthetic_sensor
from src.model.calibration_trainer import train_all as cal_train_all
from src.model.forecasting_trainer_ml import train_all_ml as fc_train_ml
from src.model.forecasting_trainer_seq import train_all_seq as fc_train_seq
from src.model.error_estimation import evaluate_calibration, evaluate_calibration_loso, evaluate_forecasting
from src.visualization.plots import save_pm25_histograms

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
FORECAST_HORIZON = 7
LOOKBACK_WINDOW = 30


def main():
    print("=" * 60)
    print("Air Quality ML Pipeline - Starting")
    print("=" * 60)

    for d in ["data/processed", "outputs/diagrams", "outputs/models", "outputs/predictions"]:
        os.makedirs(d, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ========== PHASE 1: Data Collection ==========
    print("\n" + "=" * 60)
    print("PHASE 1: Data Collection")
    print("=" * 60)

    df = collect_all_stations_data()
    df.to_csv("data/processed/all_stations.csv", index=False)
    print("Collected station data saved to: data/processed/all_stations.csv")

    # ========== PHASE 2: Weather ==========
    print("\n" + "=" * 60)
    print("PHASE 2: Adding Weather Features")
    print("=" * 60)

    df = add_weather_columns(df)
    df.to_csv("data/processed/all_stations_weather.csv", index=False)
    print("Weather-merged data saved.")

    # ========== PHASE 3: Preprocessing ==========
    print("\n" + "=" * 60)
    print("PHASE 3: Preprocessing")
    print("=" * 60)

    print("Dropping rows with all-NaN key columns...")
    df = df.dropna(subset=["PM2.5", "temperature", "humidity", "wind_speed", "pressure"], how="all")

    print("Extracting time features...")
    df = extract_time_features(df)

    df = filter_dataframe(df, pm25_max=500, min_stations=2)

    print("Selecting best features...")
    df_filtered = select_features(df)

    before_filename = f"data/processed/before_normalization_{timestamp}.csv"
    df_filtered.to_csv(before_filename, index=False)
    print("Filtered features saved to:", before_filename)

    # ========== PHASE 4: Normalization ==========
    print("\n" + "=" * 60)
    print("PHASE 4: Normalization")
    print("=" * 60)

    non_feature_columns = ['date', 'station_name', 'latitude', 'longitude']
    feature_cols = []
    for col in df_filtered.columns:
        if col not in non_feature_columns:
            feature_cols.append(col)

    df_normalized = normalize_features(df_filtered, columns_to_scale=feature_cols)
    save_pm25_histograms(df_raw=df_filtered, df_standard=df_normalized)

    after_filename = f"data/processed/after_normalization_{timestamp}.csv"
    df_normalized.to_csv(after_filename, index=False)
    print("Normalized features saved to:", after_filename)

    # ========== PHASE 5: Synthetic Sensor ==========
    print("\n" + "=" * 60)
    print("PHASE 5: Synthetic Sensor Generation")
    print("=" * 60)

    df_with_synthetic = generate_synthetic_sensor(df)
    df_with_synthetic.to_csv("data/processed/synthetic_sensor_data.csv", index=False)

    # ========== PHASE 6: Calibration Dataset Prep ==========
    print("\n" + "=" * 60)
    print("PHASE 6: Calibration Training & Evaluation")
    print("=" * 60)

    cal_df = df_with_synthetic.copy()
    cal_df["date"] = pd.to_datetime(cal_df["date"])
    cal_drop_cols = [c for c in ["PM2.5", "synthetic_pm25", "synthetic_pm10"] if c in cal_df.columns]
    cal_df = cal_df.dropna(subset=cal_drop_cols)

    cal_df = cal_df.sort_values(["station_name", "date"]).reset_index(drop=True)
    cal_df["month"] = cal_df["date"].dt.month
    cal_df["day_of_week"] = cal_df["date"].dt.dayofweek

    cal_feature_names = [
        "synthetic_pm25", "synthetic_pm10",
        "humidity", "temperature", "pressure", "wind_speed",
        "month", "day_of_week",
    ]

    cal_available = [f for f in cal_feature_names if f in cal_df.columns]
    cal_df = cal_df.dropna(subset=cal_available + ["PM2.5"])

    X_cal = cal_df[cal_available].values
    y_cal = cal_df["PM2.5"].values
    dates_cal = cal_df["date"].values

    split_idx = int(len(X_cal) * TRAIN_RATIO)
    X_train_cal, X_test_cal = X_cal[:split_idx], X_cal[split_idx:]
    y_train_cal, y_test_cal = y_cal[:split_idx], y_cal[split_idx:]
    dates_train_cal, dates_test_cal = dates_cal[:split_idx], dates_cal[split_idx:]

    cal_scaler = StandardScaler()
    X_train_cal = cal_scaler.fit_transform(X_train_cal)
    X_test_cal = cal_scaler.transform(X_test_cal)

    cal_df.to_csv("data/processed/calibration_data.csv", index=False)
    print(f"Calibration: {len(X_train_cal)} train / {len(X_test_cal)} test, {len(cal_available)} features")

    cal_models, ann_history = cal_train_all(X_train_cal, y_train_cal, cal_scaler)

    cal_results, cal_predictions = evaluate_calibration(
        cal_models, X_test_cal, y_test_cal,
        cal_available, dates_test_cal, ann_history,
    )

    print("\nCalibration Results:")
    print(cal_results.to_string(index=False))

    # ========== PHASE 6b: LOSO Calibration Validation ==========
    evaluate_calibration_loso(cal_df, cal_available)

    # ========== PHASE 7: Forecasting ML Dataset Prep ==========
    print("\n" + "=" * 60)
    print("PHASE 7: Forecasting ML Training & Evaluation")
    print("=" * 60)

    fc_df = df_filtered.copy()
    fc_df["date"] = pd.to_datetime(fc_df["date"])

    weather_cols = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
    available_weather = [c for c in weather_cols if c in fc_df.columns]

    fc_df = fc_df.dropna(subset=["PM2.5"] + available_weather)
    fc_df = fc_df.sort_values(["station_name", "date"]).reset_index(drop=True)

    all_fc_ml_dfs = []
    for station_name, group in fc_df.groupby("station_name"):
        group = group.sort_values("date").reset_index(drop=True)

        for lag in range(1, LOOKBACK_WINDOW + 1):
            group[f"pm25_lag_{lag}"] = group["PM2.5"].shift(lag)

        for window in [7, 14, 30]:
            group[f"rolling_mean_{window}"] = group["PM2.5"].shift(1).rolling(window).mean()
        group["rolling_std_7"] = group["PM2.5"].shift(1).rolling(7).std()

        group["month"] = group["date"].dt.month
        group["day_of_week"] = group["date"].dt.dayofweek
        group["day_of_year"] = group["date"].dt.dayofyear

        for h in range(1, FORECAST_HORIZON + 1):
            group[f"target_day_{h}"] = group["PM2.5"].shift(-h)

        all_fc_ml_dfs.append(group)

    fc_ml = pd.concat(all_fc_ml_dfs, ignore_index=True)
    fc_ml = fc_ml.dropna().reset_index(drop=True)

    if len(fc_ml) < 50:
        print(f"WARNING: Insufficient data for forecasting ML: {len(fc_ml)} rows (need >= 50)")
    else:
        target_cols = [f"target_day_{h}" for h in range(1, FORECAST_HORIZON + 1)]
        feature_cols_ml = [c for c in fc_ml.columns if c not in ["date", "PM2.5", "station_name", "latitude", "longitude"] + target_cols]

        X_fc_ml = fc_ml[feature_cols_ml].values
        y_fc_ml = fc_ml[target_cols].values
        dates_fc_ml = fc_ml["date"].values

        split_idx = int(len(X_fc_ml) * TRAIN_RATIO)
        X_train_fc_ml, X_test_fc_ml = X_fc_ml[:split_idx], X_fc_ml[split_idx:]
        y_train_fc_ml, y_test_fc_ml = y_fc_ml[:split_idx], y_fc_ml[split_idx:]
        dates_train_fc_ml, dates_test_fc_ml = dates_fc_ml[:split_idx], dates_fc_ml[split_idx:]

        fc_ml_scaler = StandardScaler()
        X_train_fc_ml = fc_ml_scaler.fit_transform(X_train_fc_ml)
        X_test_fc_ml = fc_ml_scaler.transform(X_test_fc_ml)

        fc_ml.to_csv("data/processed/forecasting_ml_data.csv", index=False)
        print(f"Forecasting ML: {len(X_train_fc_ml)} train / {len(X_test_fc_ml)} test, {X_fc_ml.shape[1]} features")

        fc_ml_models = fc_train_ml(X_train_fc_ml, y_train_fc_ml, fc_ml_scaler)

    # ========== PHASE 8: Forecasting Sequential Dataset Prep ==========
    print("\n" + "=" * 60)
    print("PHASE 8: Forecasting Sequential Training & Evaluation")
    print("=" * 60)

    seq_features = ["PM2.5"] + available_weather
    fc_seq_df = fc_df.copy()

    fc_seq_df["month_sin"] = np.sin(2 * np.pi * fc_seq_df["date"].dt.month / 12)
    fc_seq_df["month_cos"] = np.cos(2 * np.pi * fc_seq_df["date"].dt.month / 12)
    fc_seq_df["day_of_year_sin"] = np.sin(2 * np.pi * fc_seq_df["date"].dt.dayofyear / 365)
    fc_seq_df["day_of_year_cos"] = np.cos(2 * np.pi * fc_seq_df["date"].dt.dayofyear / 365)

    seq_feature_cols = seq_features + ["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]
    fc_seq_df = fc_seq_df.dropna(subset=seq_feature_cols)

    X_list, y_list, date_list = [], [], []

    # Fit ONE global scaler on all stations' data
    all_seq_data = fc_seq_df[seq_feature_cols].values
    seq_scaler = StandardScaler()
    seq_scaler.fit(all_seq_data)

    for station_name, group in fc_seq_df.groupby("station_name"):
        group = group.sort_values("date").reset_index(drop=True)
        data = group[seq_feature_cols].values
        dates_seq = group["date"].values

        data_scaled = seq_scaler.transform(data)

        pm25_idx = 0
        for i in range(LOOKBACK_WINDOW, len(data_scaled) - FORECAST_HORIZON):
            X_list.append(data_scaled[i - LOOKBACK_WINDOW:i])
            y_list.append(data_scaled[i:i + FORECAST_HORIZON, pm25_idx])
            date_list.append(dates_seq[i])

    if len(X_list) < 30:
        print(f"WARNING: Insufficient data for sequential forecasting: {len(X_list)} sequences (need >= 30)")
    else:
        X_seq = np.array(X_list)
        y_seq = np.array(y_list)
        dates_seq_arr = np.array(date_list)

        split_idx = int(len(X_seq) * TRAIN_RATIO)
        X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
        y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
        dates_train_seq, dates_test_seq = dates_seq_arr[:split_idx], dates_seq_arr[split_idx:]

        np.savez(
            "data/processed/forecasting_seq_data.npz",
            X_train=X_train_seq, X_test=X_test_seq,
            y_train=y_train_seq, y_test=y_test_seq,
        )
        print(f"Forecasting seq: X_train={X_train_seq.shape}, X_test={X_test_seq.shape}")

        fc_seq_models, fc_seq_histories = fc_train_seq(X_train_seq, y_train_seq, seq_scaler)

    # ========== PHASE 9: Forecasting Evaluation ==========
    print("\n" + "=" * 60)
    print("PHASE 9: Forecasting Evaluation")
    print("=" * 60)

    if len(fc_ml) >= 50 and len(X_list) >= 30:
        fc_results, fc_horizon, fc_predictions = evaluate_forecasting(
            fc_ml_models, fc_seq_models,
            (X_test_fc_ml, y_test_fc_ml, dates_test_fc_ml),
            (X_test_seq, y_test_seq, dates_test_seq),
            feature_cols_ml, fc_seq_histories, seq_scaler,
        )

        print("\nForecasting Results:")
        print(fc_results.to_string(index=False))
        print("\nPer-Horizon RMSE:")
        print(fc_horizon.to_string(index=False))
    else:
        print("Skipping forecasting evaluation due to insufficient data.")

    # ========== DONE ==========
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Results saved to: outputs/")
    print("Models saved to: outputs/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

SENSOR_BIAS = (0.15, 0.30)
SENSOR_NOISE_STD = 0.15
SENSOR_HUMIDITY_THRESHOLD = 70


def generate_synthetic_sensor(df, seed=42):
    print("Generating synthetic PMS5003 sensor data...")
    np.random.seed(seed)

    result_dfs = []

    for station_name, group in df.groupby("station_name"):
        group = group.copy()
        n = len(group)

        if "humidity" not in group.columns:
            group["humidity"] = 65.0
            print(f"No humidity data for {station_name}, using default=65%")

        for pm_col, synth_col in [("PM2.5", "synthetic_pm25"), ("PM10", "synthetic_pm10")]:
            if pm_col not in group.columns:
                print(f"Column {pm_col} not found for {station_name}, skipping.")
                continue

            ref = group[pm_col].values.astype(float)

            bias_factors = np.random.uniform(1.0 + SENSOR_BIAS[0], 1.0 + SENSOR_BIAS[1], size=n)
            noise = np.random.normal(0, SENSOR_NOISE_STD, size=n) * ref

            humidity_drift = np.zeros(n)
            high_rh_mask = group["humidity"].values > SENSOR_HUMIDITY_THRESHOLD
            humidity_drift[high_rh_mask] = ref[high_rh_mask] * np.random.uniform(0.10, 0.20, size=high_rh_mask.sum())

            synthetic = ref * bias_factors + noise + humidity_drift
            synthetic = np.maximum(synthetic, 0)
            group[synth_col] = np.round(synthetic, 2)

        result_dfs.append(group)
        print(f"Synthetic sensor generated for: {station_name} ({n} rows)")

    result = pd.concat(result_dfs, ignore_index=True)
    print(f"Total synthetic sensor data: {len(result)} rows")
    return result

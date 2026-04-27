import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from src.visualization.plots import (
    plot_scatter, plot_loss_curves,
    plot_feature_importance, plot_time_series, plot_distribution,
    plot_residuals, plot_forecast_horizon, plot_forecast_example,
)

EPA_R2_TARGET = 0.70
EPA_RMSE_TARGET = 7.0
FORECAST_HORIZON = 7
LOSO_ANN_EPOCHS = 50
LOSO_BATCH_SIZE = 32
RANDOM_SEED = 42


def evaluate_calibration(models, X_test, y_test, feature_names, dates_test, ann_history):
    print("Evaluating calibration models...")
    output_dir = os.path.join("outputs", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    predictions = {}

    for name, model in models.items():
        if name == "ANN":
            y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        epa_pass = r2 >= EPA_R2_TARGET and rmse <= EPA_RMSE_TARGET

        results.append({
            "Model": name, "R2": round(r2, 4), "RMSE": round(rmse, 4),
            "MAE": round(mae, 4), "EPA_Pass": epa_pass,
        })
        predictions[name] = y_pred
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, EPA={'PASS' if epa_pass else 'FAIL'}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join("outputs", "calibration_results.csv"), index=False)

    best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
    best_pred = predictions[best_name]
    print(f"Best calibration model: {best_name}")

    plot_scatter(
        y_test, best_pred, best_name,
        os.path.join(output_dir, "cal_scatter_best.png")
    )

    if ann_history is not None:
        plot_loss_curves(
            ann_history.history, "ANN",
            os.path.join(output_dir, "cal_ann_loss.png")
        )

    for name in ["RF", "XGBoost"]:
        if name in models:
            model = models[name]
            importances = model.feature_importances_
            plot_feature_importance(
                feature_names, importances, f"{name} Feature Importance (Calibration)",
                os.path.join(output_dir, f"cal_{name.lower()}_importance.png")
            )

    plot_time_series(
        dates_test, y_test, best_pred, best_name,
        os.path.join(output_dir, "cal_time_series.png")
    )

    plot_distribution(
        y_test, best_pred, best_name,
        os.path.join(output_dir, "cal_distribution.png")
    )

    plot_residuals(
        y_test, best_pred, best_name,
        os.path.join(output_dir, "cal_residuals.png")
    )

    return results_df, predictions


def _build_loso_ann(n_features):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(RANDOM_SEED)
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(n_features,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def evaluate_calibration_loso(cal_df, feature_names):
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    print("\n" + "=" * 60)
    print("PHASE 6b: LOSO Calibration Validation")
    print("=" * 60)

    unique_stations = cal_df["station_name"].unique()
    print(f"Stations for LOSO: {list(unique_stations)} ({len(unique_stations)} folds)")

    model_specs = {
        "MLR": lambda: LinearRegression(),
        "SVR": lambda: SVR(C=10, epsilon=0.1, kernel="rbf"),
        "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED),
        "XGBoost": lambda: XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED, verbosity=0),
    }

    results = []

    for held_out in unique_stations:
        print(f"\n--- Fold: held-out = {held_out} ---")
        train_mask = cal_df["station_name"] != held_out
        test_mask = cal_df["station_name"] == held_out

        X_train = cal_df[train_mask][feature_names].values
        y_train = cal_df[train_mask]["PM2.5"].values
        X_test = cal_df[test_mask][feature_names].values
        y_test = cal_df[test_mask]["PM2.5"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        for name, make_model in model_specs.items():
            model = make_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            results.append({"Model": name, "Fold": held_out, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
            print(f"  {name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        # ANN
        ann = _build_loso_ann(X_train.shape[1])
        val_size = int(len(X_train) * 0.15)
        ann.fit(
            X_train[:-val_size], y_train[:-val_size],
            validation_data=(X_train[-val_size:], y_train[-val_size:]),
            epochs=LOSO_ANN_EPOCHS, batch_size=LOSO_BATCH_SIZE, verbose=0,
        )
        y_pred = ann.predict(X_test, verbose=0).flatten()
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Model": "ANN", "Fold": held_out, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        print(f"  ANN: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    results_df = pd.DataFrame(results)

    # Compute mean per model across folds
    mean_rows = []
    for model_name in ["MLR", "SVR", "RF", "XGBoost", "ANN"]:
        subset = results_df[results_df["Model"] == model_name]
        mean_rows.append({
            "Model": model_name, "Fold": "Mean",
            "R2": round(subset["R2"].mean(), 4),
            "RMSE": round(subset["RMSE"].mean(), 4),
            "MAE": round(subset["MAE"].mean(), 4),
        })

    results_df = pd.concat([results_df, pd.DataFrame(mean_rows)], ignore_index=True)

    out_path = os.path.join("outputs", "calibration_loso_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nLOSO results saved to: {out_path}")
    print("\nLOSO Calibration Results:")
    print(results_df.to_string(index=False))

    return results_df


def _compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return r2, rmse, mae


def _per_horizon_rmse(y_true, y_pred):
    rmses = []
    for h in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        rmses.append(rmse)
    return rmses


def evaluate_forecasting(ml_models, seq_models, ml_data, seq_data, feature_names_ml, seq_histories, seq_scaler):
    print("Evaluating forecasting models...")
    output_dir = os.path.join("outputs", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    X_test_ml, y_test_ml, dates_test_ml = ml_data
    X_test_seq, y_test_seq, dates_test_seq = seq_data

    # Extract PM2.5 scaler params for inverse transform (PM2.5 is column 0)
    pm25_mean = seq_scaler.mean_[0]
    pm25_scale = seq_scaler.scale_[0]

    all_results = []
    all_horizon_rmses = {}
    all_predictions = {}

    for name, model in ml_models.items():
        y_pred = model.predict(X_test_ml)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, FORECAST_HORIZON)

        r2, rmse, mae = _compute_metrics(y_test_ml, y_pred)
        horizon_rmses = _per_horizon_rmse(y_test_ml, y_pred)

        all_results.append({"Model": name, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        all_horizon_rmses[name] = horizon_rmses
        all_predictions[name] = (y_test_ml, y_pred, dates_test_ml)
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    for name, model in seq_models.items():
        y_pred_scaled = model.predict(X_test_seq, verbose=0)
        # Inverse transform from normalized scale to original µg/m³
        y_pred = y_pred_scaled * pm25_scale + pm25_mean
        y_test = y_test_seq * pm25_scale + pm25_mean

        r2, rmse, mae = _compute_metrics(y_test, y_pred)
        horizon_rmses = _per_horizon_rmse(y_test, y_pred)

        all_results.append({"Model": name, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        all_horizon_rmses[name] = horizon_rmses
        all_predictions[name] = (y_test, y_pred, dates_test_seq)
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join("outputs", "forecasting_results.csv"), index=False)

    horizon_data = []
    for name, rmses in all_horizon_rmses.items():
        for h, rmse_val in enumerate(rmses, 1):
            horizon_data.append({"Model": name, "Horizon_Day": h, "RMSE": round(rmse_val, 4)})
    horizon_df = pd.DataFrame(horizon_data)
    horizon_df.to_csv(os.path.join("outputs", "forecasting_per_horizon.csv"), index=False)

    best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
    best_y_true, best_y_pred, best_dates = all_predictions[best_name]
    print(f"Best forecasting model: {best_name}")

    plot_scatter(
        best_y_true.flatten(), best_y_pred.flatten(), best_name,
        os.path.join(output_dir, "fc_scatter_best.png")
    )

    for name, history in seq_histories.items():
        safe_name = name.lower().replace("-", "_")
        plot_loss_curves(
            history.history, name,
            os.path.join(output_dir, f"fc_{safe_name}_loss.png")
        )

    for name in ["RF", "XGBoost"]:
        if name in ml_models:
            model = ml_models[name]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "estimators_"):
                importances = np.mean(
                    [est.feature_importances_ for est in model.estimators_], axis=0
                )
            else:
                continue
            plot_feature_importance(
                feature_names_ml, importances, f"{name} Feature Importance (Forecasting)",
                os.path.join(output_dir, f"fc_{name.lower()}_importance.png")
            )

    plot_time_series(
        best_dates, best_y_true[:, 0], best_y_pred[:, 0],
        f"{best_name} (Day 1)",
        os.path.join(output_dir, "fc_time_series.png")
    )

    plot_distribution(
        best_y_true.flatten(), best_y_pred.flatten(), best_name,
        os.path.join(output_dir, "fc_distribution.png")
    )

    plot_residuals(
        best_y_true.flatten(), best_y_pred.flatten(), best_name,
        os.path.join(output_dir, "fc_residuals.png")
    )

    plot_forecast_horizon(
        all_horizon_rmses,
        os.path.join(output_dir, "fc_rmse_horizon.png")
    )

    n_samples = min(3, len(best_y_true))
    np.random.seed(42)
    sample_indices = np.random.choice(len(best_y_true), n_samples, replace=False)
    for i, idx in enumerate(sorted(sample_indices)):
        plot_forecast_example(
            best_y_true[idx], best_y_pred[idx], best_name, idx,
            os.path.join(output_dir, f"fc_example_{i+1}.png")
        )

    return results_df, horizon_df, all_predictions

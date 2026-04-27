import os
import numpy as np
import joblib

RANDOM_SEED = 42
N_SPLITS_CV = 5
FORECAST_HORIZON = 7


def train_mlr(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor
    print("Training Forecasting MLR...")
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("outputs", "models", "forecasting_mlr.joblib"))
    return model


def train_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    print("Training Forecasting RF...")

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED),
        param_dist, n_iter=30, cv=tscv, scoring="neg_mean_squared_error",
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train)

    print(f"RF best params: {search.best_params_}")
    joblib.dump(search.best_estimator_, os.path.join("outputs", "models", "forecasting_rf.joblib"))
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    print("Training Forecasting XGBoost...")

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    base = XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=50, cv=tscv, scoring="neg_mean_squared_error",
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train[:, 0])

    best_params = search.best_params_
    print(f"XGBoost best params: {best_params}")

    model = MultiOutputRegressor(XGBRegressor(**best_params, random_state=RANDOM_SEED, verbosity=0))
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join("outputs", "models", "forecasting_xgb.joblib"))
    return model


def train_all_ml(X_train, y_train, scaler):
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join("outputs", "models", "forecasting_scaler_ml.joblib"))

    models = {}
    models["MLR"] = train_mlr(X_train, y_train)
    models["RF"] = train_rf(X_train, y_train)
    models["XGBoost"] = train_xgboost(X_train, y_train)

    print("All forecasting ML models trained.")
    return models

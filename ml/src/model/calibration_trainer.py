import os
import numpy as np
import joblib

RANDOM_SEED = 42
N_SPLITS_CV = 5
EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10


def train_mlr(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    print("Training MLR...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("outputs", "models", "calibration_mlr.joblib"))
    return model


def train_svr(X_train, y_train):
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    print("Training SVR with GridSearchCV...")

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["rbf"],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    grid = GridSearchCV(SVR(), param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"SVR best params: {grid.best_params_}")
    joblib.dump(grid.best_estimator_, os.path.join("outputs", "models", "calibration_svr.joblib"))
    return grid.best_estimator_


def train_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    print("Training RF with RandomizedSearchCV...")

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
    joblib.dump(search.best_estimator_, os.path.join("outputs", "models", "calibration_rf.joblib"))
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    print("Training XGBoost with RandomizedSearchCV...")

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_SEED, verbosity=0),
        param_dist, n_iter=50, cv=tscv, scoring="neg_mean_squared_error",
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train)

    print(f"XGBoost best params: {search.best_params_}")
    joblib.dump(search.best_estimator_, os.path.join("outputs", "models", "calibration_xgb.joblib"))
    return search.best_estimator_


def train_ann(X_train, y_train):
    import tensorflow as tf
    from tensorflow import keras
    print("Training ANN...")

    tf.random.set_seed(RANDOM_SEED)

    val_size = int(len(X_train) * 0.15)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )

    model.save(os.path.join("outputs", "models", "calibration_ann.keras"))
    print(f"ANN trained: {len(history.history['loss'])} epochs")
    return model, history


def train_all(X_train, y_train, scaler):
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join("outputs", "models", "calibration_scaler.joblib"))

    models = {}
    models["MLR"] = train_mlr(X_train, y_train)
    models["SVR"] = train_svr(X_train, y_train)
    models["RF"] = train_rf(X_train, y_train)
    models["XGBoost"] = train_xgboost(X_train, y_train)

    ann_model, ann_history = train_ann(X_train, y_train)
    models["ANN"] = ann_model

    print("All calibration models trained.")
    return models, ann_history

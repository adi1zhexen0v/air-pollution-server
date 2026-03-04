import os
import numpy as np
import joblib

RANDOM_SEED = 42
FORECAST_HORIZON = 7
EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10


def train_lstm(X_train, y_train, X_val, y_val):
    import tensorflow as tf
    from tensorflow import keras
    print("Training Forecasting LSTM...")

    tf.random.set_seed(RANDOM_SEED)

    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(FORECAST_HORIZON),
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )

    model.save(os.path.join("outputs", "models", "forecasting_lstm.keras"))
    print(f"LSTM trained: {len(history.history['loss'])} epochs")
    return model, history


def train_cnn_lstm(X_train, y_train, X_val, y_val):
    import tensorflow as tf
    from tensorflow import keras
    print("Training Forecasting CNN-LSTM...")

    tf.random.set_seed(RANDOM_SEED)

    model = keras.Sequential([
        keras.layers.Conv1D(64, kernel_size=3, activation="relu",
                            input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(FORECAST_HORIZON),
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )

    model.save(os.path.join("outputs", "models", "forecasting_cnn_lstm.keras"))
    print(f"CNN-LSTM trained: {len(history.history['loss'])} epochs")
    return model, history


def train_all_seq(X_train, y_train, scaler):
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join("outputs", "models", "forecasting_scaler_seq.joblib"))

    val_size = int(len(X_train) * 0.15)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

    models = {}
    histories = {}

    lstm_model, lstm_history = train_lstm(X_tr, y_tr, X_val, y_val)
    models["LSTM"] = lstm_model
    histories["LSTM"] = lstm_history

    cnn_lstm_model, cnn_lstm_history = train_cnn_lstm(X_tr, y_tr, X_val, y_val)
    models["CNN-LSTM"] = cnn_lstm_model
    histories["CNN-LSTM"] = cnn_lstm_history

    print("All forecasting sequential models trained.")
    return models, histories

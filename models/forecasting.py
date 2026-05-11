# models/forecasting.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# =========================
# DATASET CREATION
# =========================
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)


# =========================
# TRAIN MODEL (SINGLE STOCK)
# =========================
def train_lstm(series, time_step=10, epochs=5):
    series = series.dropna().values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = create_dataset(scaled, time_step)

    if len(X) < 20:
        return None  # not enough data

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape(X_train.shape[0], time_step, 1)
    X_test = X_test.reshape(X_test.shape[0], time_step, 1)

    model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    pred = model.predict(X_test, verbose=0)

    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    last_seq = scaled[-time_step:]

    return {
        "model": model,
        "scaler": scaler,
        "last_seq": last_seq,
        "y_test": y_test.flatten(),
        "pred": pred.flatten(),
        "mae": mae,
        "rmse": rmse,
        "history": history
    }


# =========================
# FUTURE FORECAST
# =========================
def forecast_future(model, last_seq, scaler, steps=30):
    future = []
    seq = last_seq.copy()

    for _ in range(steps):
        pred = model.predict(seq.reshape(1, len(seq), 1), verbose=0)
        future.append(pred[0, 0])
        seq = np.append(seq[1:], pred[0, 0])

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))
    return future.flatten()
"""Shared utilities for the selection pipeline."""

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42

# EPA PM2.5 AQI breakpoint table:
# (AQI_lo, AQI_hi, conc_lo, conc_hi)
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
    """Convert a US EPA PM2.5 AQI index value to ug/m3 concentration.

    Uses the standard EPA breakpoint linear interpolation formula:
        C = (C_hi - C_lo) / (I_hi - I_lo) * (I - I_lo) + C_lo
    """
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return None
    aqi = float(aqi)
    if aqi < 0:
        return 0.0
    for aqi_lo, aqi_hi, conc_lo, conc_hi in _PM25_BREAKPOINTS:
        if aqi_lo <= aqi <= aqi_hi:
            return (conc_hi - conc_lo) / (aqi_hi - aqi_lo) * (aqi - aqi_lo) + conc_lo
    # AQI > 500: extrapolate from last segment
    aqi_lo, aqi_hi, conc_lo, conc_hi = _PM25_BREAKPOINTS[-1]
    return (conc_hi - conc_lo) / (aqi_hi - aqi_lo) * (aqi - aqi_lo) + conc_lo


def compute_heat_index(temperature_c, humidity_pct):
    """Compute heat index using the NOAA formula (matching ESP32 DHT library).

    Parameters
    ----------
    temperature_c : float
        Temperature in Celsius.
    humidity_pct : float
        Relative humidity in percent (0-100).

    Returns
    -------
    float
        Heat index in Celsius.
    """
    if temperature_c is None or humidity_pct is None:
        return None
    if isinstance(temperature_c, float) and math.isnan(temperature_c):
        return None
    if isinstance(humidity_pct, float) and math.isnan(humidity_pct):
        return None

    # Convert to Fahrenheit for the formula
    t = temperature_c * 9.0 / 5.0 + 32.0
    rh = float(humidity_pct)

    # Simple formula for low temperatures
    hi = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)

    if hi < 80:
        return (hi - 32.0) * 5.0 / 9.0

    # Full Rothfusz regression
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t * t
        - 0.05481717 * rh * rh
        + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh
        - 0.00000199 * t * t * rh * rh
    )

    # Adjustments
    if rh < 13 and 80 <= t <= 112:
        hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
    elif rh > 85 and 80 <= t <= 87:
        hi += ((rh - 85) / 10) * ((87 - t) / 5)

    return (hi - 32.0) * 5.0 / 9.0


def compute_metrics(y_true, y_pred):
    """Compute R2, RMSE, and MAE.

    Returns
    -------
    dict
        {"r2": float, "rmse": float, "mae": float}
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

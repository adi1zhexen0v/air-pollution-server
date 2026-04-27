import numpy as np
import pandas as pd

# US EPA AQI breakpoints for PM2.5 (ug/m3, 24-hr)
PM25_BREAKPOINTS = [
    (0,   50,   0.0,  12.0),
    (51,  100,  12.1, 35.4),
    (101, 150,  35.5, 55.4),
    (151, 200,  55.5, 150.4),
    (201, 300,  150.5, 250.4),
    (301, 400,  250.5, 350.4),
    (401, 500,  350.5, 500.4),
]

# US EPA AQI breakpoints for PM10 (ug/m3, 24-hr)
PM10_BREAKPOINTS = [
    (0,   50,   0,   54),
    (51,  100,  55,  154),
    (101, 150,  155, 254),
    (151, 200,  255, 354),
    (201, 300,  355, 424),
    (301, 400,  425, 504),
    (401, 500,  505, 604),
]


def _aqi_to_concentration(aqi_value, breakpoints):
    if pd.isna(aqi_value):
        return np.nan
    aqi_value = float(aqi_value)
    if aqi_value < 0:
        return np.nan
    aqi_value = min(aqi_value, 500)

    for aqi_lo, aqi_hi, conc_lo, conc_hi in breakpoints:
        if aqi_lo <= aqi_value <= aqi_hi:
            concentration = ((conc_hi - conc_lo) / (aqi_hi - aqi_lo)) * (aqi_value - aqi_lo) + conc_lo
            return round(concentration, 1)

    return np.nan


def aqi_to_pm25(aqi):
    return _aqi_to_concentration(aqi, PM25_BREAKPOINTS)


def aqi_to_pm10(aqi):
    return _aqi_to_concentration(aqi, PM10_BREAKPOINTS)

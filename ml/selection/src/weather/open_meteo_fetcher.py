import time
import requests
import pandas as pd

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RETRIES = 3
RETRY_DELAY = 5


def get_daily_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
            "precipitation_sum",
        ]),
        "timezone": "Asia/Almaty",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching weather for ({lat}, {lon}) attempt {attempt}/{MAX_RETRIES}...")
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed to fetch weather after {MAX_RETRIES} attempts") from e
            time.sleep(RETRY_DELAY)

    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "pressure": daily["surface_pressure_mean"],
        "wind_speed": daily["wind_speed_10m_mean"],
        "precipitation": daily["precipitation_sum"],
    })

    df = df.set_index("date").asfreq("D")
    df = df.ffill(limit=1)
    df = df.dropna().reset_index()
    df["latitude"] = lat
    df["longitude"] = lon

    return df


def add_weather_columns(df):
    df["date"] = pd.to_datetime(df["date"])
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    stations = df[["station_name", "latitude", "longitude"]].drop_duplicates()
    weather_all = []

    station_counter = 1
    total_stations = len(stations)

    for _, station in stations.iterrows():
        print(f"Fetching weather for {station['station_name']} ({station_counter}/{total_stations})")
        station_counter += 1

        time.sleep(15)

        daily_weather = get_daily_weather(
            lat=station["latitude"],
            lon=station["longitude"],
            start_date=start_date,
            end_date=end_date
        )
        daily_weather["station_name"] = station["station_name"]
        weather_all.append(daily_weather)

    weather_df = pd.concat(weather_all, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    merged_df = pd.merge(
        df,
        weather_df,
        on=["date", "station_name", "latitude", "longitude"],
        how="left"
    )

    merged_df["date"] = pd.to_datetime(merged_df["date"])

    return merged_df

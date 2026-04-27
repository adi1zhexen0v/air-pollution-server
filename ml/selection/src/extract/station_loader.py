import pandas as pd
from data.stations import stations
from src.preprocess.aqi_converter import aqi_to_pm25, aqi_to_pm10


def load_embassy_data(file_path, station_name, lat, lon):
    print(f"Loading embassy data from: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df['date'] = pd.to_datetime(df['date'].str.strip(), format='%Y/%m/%d')
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
    df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')

    df['PM2.5'] = df['pm25'].apply(aqi_to_pm25)
    df['PM10'] = df['pm10'].apply(aqi_to_pm10)

    result = df[['date', 'PM2.5', 'PM10']].copy()
    result = result.dropna(subset=['PM2.5'])
    result = result.sort_values('date').drop_duplicates(subset=['date'], keep='first').reset_index(drop=True)

    result['station_name'] = station_name
    result['latitude'] = lat
    result['longitude'] = lon

    result['date'] = result['date'].dt.date

    print(f"Embassy data loaded: {len(result)} rows")
    return result


def load_kazhydromet_data(files_dict, station_name, lat, lon):
    print(f"Loading Kazhydromet data for: {station_name}")
    param_dfs = {}

    for param_name, file_path in files_dict.items():
        try:
            df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if 'date' in df.columns and 'median' in df.columns:
            col_name = 'PM2.5' if param_name == 'pm25' else 'PM10'
            df = df[['date', 'median']].copy()
            df.rename(columns={'median': col_name}, inplace=True)
            param_dfs[col_name] = df
        else:
            print(f"File '{file_path}' skipped: missing 'date' or 'median' column.")

    if len(param_dfs) == 0:
        print(f"No valid files for station '{station_name}'.")
        return pd.DataFrame()

    merged_df = None
    for param_name in param_dfs:
        if merged_df is None:
            merged_df = param_dfs[param_name]
        else:
            merged_df = pd.merge(merged_df, param_dfs[param_name], on='date', how='outer')

    merged_df['station_name'] = station_name
    merged_df['latitude'] = lat
    merged_df['longitude'] = lon

    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.drop_duplicates(subset=['date', 'station_name'])
    merged_df = merged_df.sort_values('date')

    print(f"Kazhydromet data loaded: {len(merged_df)} rows")
    return merged_df


def collect_all_stations_data():
    all_stations_data = []

    for station in stations:
        if station['type'] == 'embassy':
            df = load_embassy_data(
                file_path=station['files']['combined'],
                station_name=station['name'],
                lat=station['lat'],
                lon=station['lon']
            )
        elif station['type'] == 'kazhydromet':
            df = load_kazhydromet_data(
                files_dict=station['files'],
                station_name=station['name'],
                lat=station['lat'],
                lon=station['lon']
            )
        else:
            print(f"Unknown station type: {station['type']}")
            continue

        if not df.empty:
            all_stations_data.append(df)
            print(f"Collected data for station: {station['name']}")
        else:
            print(f"No data collected for station: {station['name']}")

    if len(all_stations_data) > 0:
        combined_df = pd.concat(all_stations_data, ignore_index=True)
        print(f"Collected data from {len(all_stations_data)} stations.")
        return combined_df
    else:
        print("No data collected from any station.")
        return pd.DataFrame()

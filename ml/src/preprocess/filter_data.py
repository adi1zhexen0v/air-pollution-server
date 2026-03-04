import pandas as pd


def filter_dataframe(df, min_station_coverage=0.5, pm25_max=500, min_stations=None):
    print("Filtering by dynamic common timeframe based on station coverage...")

    df = df[(df["PM2.5"] >= 0) & (df["PM2.5"] <= pm25_max)]
    print("Filtered PM2.5 values: kept within 0 to", pm25_max)

    total_station_count = df["station_name"].nunique()
    if min_stations is not None:
        min_required_stations = min_stations
    else:
        min_required_stations = int(total_station_count * min_station_coverage)

    station_counts_by_date = df.groupby("date")["station_name"].nunique()
    valid_dates = station_counts_by_date[station_counts_by_date >= min_required_stations].index

    if len(valid_dates) == 0:
        print("No dates meet the required station coverage. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    first_valid_date = valid_dates.min().strftime('%Y-%m-%d')
    last_valid_date = valid_dates.max().strftime('%Y-%m-%d')

    print("Selected", len(valid_dates), "valid dates from", first_valid_date, "to", last_valid_date)
    print("With at least", min_required_stations, "out of", total_station_count, "stations per date")

    df_filtered = df[df["date"].isin(valid_dates)].copy()

    return df_filtered

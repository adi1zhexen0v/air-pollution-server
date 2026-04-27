"""Prepare multi-station daily CSV for forecasting_selection.py.

Reads all_stations_weather.csv, filters to dates with >= min_stations,
fills short gaps per station, and outputs data sorted by (station, date).

Usage:
    python prepare_forecasting_data.py                           # all stations, min 2
    python prepare_forecasting_data.py --min-stations 1          # all dates
    python prepare_forecasting_data.py --station "US Embassy"    # single station mode
    python prepare_forecasting_data.py --list-stations
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import aqi_to_ugm3_pm25

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), "data", "processed", "all_stations_weather.csv"
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "forecasting_ready.csv")


def fill_gaps_per_station(df, max_gap=7):
    """Reindex each station to a full date range, interpolate gaps <= max_gap days."""
    filled_parts = []
    for station, sdf in df.groupby("station_name"):
        sdf = sdf.sort_values("date").set_index("date")
        full_range = pd.date_range(sdf.index.min(), sdf.index.max(), freq="D")
        total_gaps = len(full_range) - len(sdf)
        sdf = sdf.reindex(full_range).rename_axis("date")
        sdf["station_name"] = station

        numeric_cols = [c for c in sdf.columns if c not in ("station_name",)]
        sdf[numeric_cols] = sdf[numeric_cols].interpolate(
            method="linear", limit=max_gap, limit_area="inside"
        )
        sdf = sdf.dropna(subset=["pm25"]).reset_index()
        if total_gaps > 0:
            filled = len(sdf) - (len(full_range) - total_gaps)
            print(f"  {station}: filled {filled} gaps (<=7d), {total_gaps - filled} dropped")
        filled_parts.append(sdf)
    return pd.concat(filled_parts, ignore_index=True)


def keep_longest_segment_per_station(df):
    """For each station, keep only the longest continuous date segment."""
    result_parts = []
    for station, sdf in df.groupby("station_name"):
        sdf = sdf.sort_values("date").reset_index(drop=True)
        diffs = sdf["date"].diff().dt.days.fillna(1)
        segments = []
        start = 0
        for i in range(1, len(sdf)):
            if diffs.iloc[i] > 1:
                segments.append((start, i))
                start = i
        segments.append((start, len(sdf)))

        best_start, best_end = max(segments, key=lambda s: s[1] - s[0])
        seg = sdf.iloc[best_start:best_end].copy()
        if len(seg) < len(sdf):
            print(f"  {station}: kept longest segment {len(seg)}/{len(sdf)} days "
                  f"({seg['date'].min().date()} to {seg['date'].max().date()})")
        else:
            print(f"  {station}: {len(seg)} continuous days")
        result_parts.append(seg)
    return pd.concat(result_parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare forecasting CSV")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Source CSV path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station mode (default: all stations)")
    parser.add_argument("--min-stations", type=int, default=2,
                        help="Keep only dates with >= N stations (default: 2)")
    parser.add_argument("--days", type=int, default=0,
                        help="Keep last N days per station (0 = all, default: 0)")
    parser.add_argument("--list-stations", action="store_true",
                        help="List available stations and exit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if args.list_stations:
        if "station_name" in df.columns:
            for name, group in df.groupby("station_name"):
                print(f"  {name}: {len(group)} rows")
        sys.exit(0)

    # Rename columns
    rename_map = {}
    if "PM2.5" in df.columns and "pm25" not in df.columns:
        rename_map["PM2.5"] = "pm25"
    if "PM10" in df.columns:
        rename_map["PM10"] = "pm10"
    if rename_map:
        df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["pm25"])

    # Convert AQI to ug/m3 per station (AQICN stations like US Embassy report AQI)
    if "station_name" in df.columns:
        for station in df["station_name"].unique():
            mask = df["station_name"] == station
            station_pm = df.loc[mask, "pm25"]
            if station_pm.max() > 300 or station_pm.mean() > 10:
                df.loc[mask, "pm25"] = station_pm.apply(aqi_to_ugm3_pm25)
                new_mean = df.loc[mask, "pm25"].mean()
                print(f"  Converted AQI->ug/m3 for {station} (new mean={new_mean:.1f})")
    elif "pm25" in df.columns and df["pm25"].max() > 300:
        df["pm25"] = df["pm25"].apply(aqi_to_ugm3_pm25)
        print(f"  Converted AQI->ug/m3 (new mean={df['pm25'].mean():.1f})")

    # Single station mode
    if args.station:
        df = df[df["station_name"] == args.station].copy()
        print(f"Filtered to '{args.station}': {len(df)} rows")
        if df.empty:
            print("ERROR: Station not found. Use --list-stations.")
            sys.exit(1)

    # Filter to dates where >= min_stations have data
    if args.min_stations > 1 and "station_name" in df.columns:
        date_counts = df.groupby("date")["station_name"].nunique()
        valid_dates = date_counts[date_counts >= args.min_stations].index
        before = len(df)
        df = df[df["date"].isin(valid_dates)].copy()
        print(f"Filtered to dates with >= {args.min_stations} stations: "
              f"{before} -> {len(df)} rows ({len(valid_dates)} dates)")

    # Filter to last N days per station
    if args.days > 0:
        parts = []
        for station, sdf in df.groupby("station_name"):
            cutoff = sdf["date"].max() - pd.Timedelta(days=args.days)
            parts.append(sdf[sdf["date"] >= cutoff])
        df = pd.concat(parts, ignore_index=True)
        print(f"Filtered to last {args.days} days per station: {len(df)} rows")

    # Keep only useful columns
    keep_cols = ["date", "station_name", "pm25"]
    optional_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols].sort_values(["station_name", "date"]).reset_index(drop=True)

    # Fill short gaps per station
    print("\nFilling date gaps per station:")
    df = fill_gaps_per_station(df)

    # Keep longest continuous segment per station
    print("\nKeeping longest continuous segment per station:")
    df = keep_longest_segment_per_station(df)

    # Sort by (station, date) for per-station sequence building
    df = df.sort_values(["station_name", "date"]).reset_index(drop=True)

    # Summary
    print(f"\nOutput: {len(df)} rows")
    stations = df["station_name"].unique()
    for s in stations:
        sub = df[df["station_name"] == s]
        print(f"  {s}: {len(sub)} days, {sub['date'].min().date()} to {sub['date'].max().date()}, "
              f"PM2.5 mean={sub['pm25'].mean():.1f}")
    print(f"  Columns: {list(df.columns)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"\nRun: python selection/forecasting_selection.py --csv {args.output}")


if __name__ == "__main__":
    main()

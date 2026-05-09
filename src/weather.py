import time
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "dallas": {"lat": 32.78, "lon": -96.81, "zone": "NCENT"},
    "houston": {"lat": 29.75, "lon": -95.36, "zone": "COAST"},
    "san_antonio": {"lat": 29.42, "lon": -98.49, "zone": "SCENT"}
}


def fetch_weather(lat, lon, start_date, end_date, station_name):
    """
    Fetch hourly weather for one location from Open-Meteo

    Returns a DataFrame with timestamp column and weather columns with station names
    """

    parameters = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago"
    }

    response = requests.get(ARCHIVE_URL, params=parameters, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        f"{station_name}_temp_f": hourly["temperature_2m"],
        f"{station_name}_humidity": hourly["relative_humidity_2m"],
        f"{station_name}_dewpoint_f": hourly["dew_point_2m"],
        f"{station_name}_wind_mph": hourly["wind_speed_10m"],
    })

    return df

def fetch_all_cities(cities, start, end):
    """
    Fetch weather for every city in cities and join on timestamp
    """
    frames = []
    for name, info in cities.items():
        print(f"Fetching {name}...")
        df = fetch_weather(info["lat"], info["lon"], start, end, name)
        frames.append(df)
        time.sleep(1)

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp", how="outer")

    return merged.sort_values("timestamp").reset_index(drop=True)

if __name__ == "__main__":
    START_DATE = "2024-01-01"
    END_DATE = "2025-12-31"

    weather = fetch_all_cities(CITIES, START_DATE, END_DATE)

    weather.to_csv(ROOT / "data/processed/weather.csv", index=False)
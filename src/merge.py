import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

LOAD_PATH = ROOT / "data/processed/combined_ercot_load.csv"
WEATHER_PATH = ROOT / "data/processed/weather.csv"
OUTPUT_PATH = ROOT / "data/processed/ercot_with_weather.csv"


def main():
    load = pd.read_csv(LOAD_PATH, parse_dates=["timestamp"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])

    load["timestamp"] = load["timestamp"] - pd.Timedelta(hours=1)

    # drop DST duplicates from each side
    load = load.drop_duplicates(subset="timestamp", keep="first")
    weather = weather.drop_duplicates(subset="timestamp", keep="first")

    merged = load.merge(weather, on="timestamp", how="left")

    merged.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
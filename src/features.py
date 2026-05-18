import numpy as np
import pandas as pd
import holidays
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMBINED_PATH = ROOT / "data/processed/ercot_with_weather.csv"
FEATURES_PATH = ROOT / "data/processed/features.csv"

BASE_TEMP = 65.0

def build_features(df):
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)

    # time
    hour = df.index.hour
    month = df.index.month
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = df.index.dayofweek
    out["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    out["month"] = month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # holidays
    year_range = range(df.index.year.min(), df.index.year.max() + 1)
    us_holidays = holidays.US(years=year_range)
    holiday_ts = pd.to_datetime(list(us_holidays.keys()))
    out["is_holiday"] = df.index.normalize().isin(holiday_ts).astype(int)

    # weather
    temp = df["dallas_temp_f"]
    out["dallas_cdd"] = (temp - BASE_TEMP).clip(lower=0)
    out["dallas_hdd"] = (BASE_TEMP - temp).clip(lower=0)
    out["dallas_humidity"] = df["dallas_humidity"]
    out["dallas_dewpoint_f"] = df["dallas_dewpoint_f"]

    # lags
    out["load_lag_24"] = df["ERCOT"].shift(24)
    out["load_lag_168"] = df["ERCOT"].shift(168)

    out["ERCOT"] = df["ERCOT"]
    out=out.dropna()

    return out


if __name__ == "__main__":
    df = pd.read_csv(COMBINED_PATH, parse_dates=["timestamp"])
    features = build_features(df)
    features.to_csv(FEATURES_PATH)
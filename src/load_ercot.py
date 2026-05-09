import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

def parse_timestamps(s):
    """
    ERCOT marks timestamps in ending hours, so convert to pd.Timestamp
    """
    s = s.strip()

    # fall back hours are marked with DST
    dst = s.endswith(" DST")
    if dst:
        s = s[:-4].strip()

    date, time = s.split(" ")
    hour = int(time.split(":")[0])

    if hour == 24:
        ts = pd.to_datetime(date, format="%m/%d/%Y") + pd.Timedelta(days=1)
    else:
        ts = pd.to_datetime(s, format="%m/%d/%Y %H:%M")

    return ts

def load_file(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=["Hour Ending"]).copy() # remove empty rows
    df["timestamp"] = df["Hour Ending"].apply(parse_timestamps)
    df = df.drop(columns=["Hour Ending"])

    # reorder so timestamp is first column
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    return df[cols].sort_values("timestamp").reset_index(drop=True)

if __name__ == "__main__":
    df_2024 = load_file(ROOT / "data/raw/Native_Load_2024.xlsx")
    df_2025 = load_file(ROOT / "data/raw/Native_Load_2025.xlsx")
    df = pd.concat([df_2024, df_2025], ignore_index=True)

    df.to_csv(ROOT / "data/processed/combined_ercot_load.csv", index=False)
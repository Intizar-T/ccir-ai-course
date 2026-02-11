"""
Merge all raw datasets into a single merged_data.csv feedable to train.py.

Sources:
  - HDHI Admission data: admissions aggregated by D.O.A (Number of Admissions, Brought_Dead_Count)
  - Raw_data 2017/2018/2019: PM2.5, PM10, NO2, Ozone, CO, RH (µg/m³ pollutants)
  - Ludhiana AQI bulletins: Index Value, Air Quality
  - temperature_data: temperature_2m_max/min, apparent_temperature_max

Strategy:
  1. Build date index from union of all sources (restrict to 2017-2019 where pollutant data exists)
  2. Left-join admissions (target) - we keep dates that have at least some feature data
  3. Merge pollutants, AQI, temperature; forward/back-fill sparse sources (AQI)
  4. Output columns match train.py expectations exactly
"""

import os
import glob
import pandas as pd
import numpy as np


# Columns required by train.py
TIMESTAMP_COL = "Timestamp"
TARGET_COL = "Number of Admissions"
FEATURE_COLS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "Index Value",
    "temperature_2m_max (°C)",
    "apparent_temperature_max (°C)",
    "Ozone (µg/m³)",
    "NO2 (µg/m³)",
    "CO (mg/m³)",
    "RH (%)",
    "temperature_2m_min (°C)",
]


def load_hdhi_admissions(path: str) -> pd.DataFrame:
    """Load HDHI admission data and parse D.O.A (Date of Admission)."""
    df = pd.read_csv(path)
    def parse_doa(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]:
            try:
                return pd.to_datetime(s, format=fmt)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    df["doa_parsed"] = df["D.O.A"].apply(parse_doa)
    df = df.dropna(subset=["doa_parsed"])
    df["date"] = df["doa_parsed"].dt.date
    return df


def aggregate_admissions(hdhi: pd.DataFrame) -> pd.DataFrame:
    """Count admissions and brought-dead per date."""
    admissions = hdhi.groupby("date").size().reset_index(name=TARGET_COL)
    if "OUTCOME" in hdhi.columns and "D.O.D" in hdhi.columns:
        same_day = hdhi.copy()
        same_day["dod_parsed"] = pd.to_datetime(same_day["D.O.D"], format="%m/%d/%Y", errors="coerce")
        mask_na = same_day["dod_parsed"].isna()
        same_day.loc[mask_na, "dod_parsed"] = pd.to_datetime(
            same_day.loc[mask_na, "D.O.D"], format="%d/%m/%Y", errors="coerce"
        )
        same_day = same_day.dropna(subset=["dod_parsed"])
        same_day["dod_date"] = same_day["dod_parsed"].dt.date
        bd_mask = (same_day["date"] == same_day["dod_date"]) & (
            same_day["OUTCOME"].astype(str).str.upper().str.contains("EXPIRY", na=False)
        )
        bd_counts = same_day[bd_mask].groupby("date").size().reset_index(name="Brought_Dead_Count")
        admissions = admissions.merge(bd_counts, on="date", how="left")
        admissions["Brought_Dead_Count"] = admissions["Brought_Dead_Count"].fillna(0).astype(int)
    else:
        admissions["Brought_Dead_Count"] = 0
    return admissions


def load_raw_pollutants(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate Raw_data_1Day_2017, 2018, 2019 (exclude duplicate 2019 files)."""
    pattern = os.path.join(raw_dir, "Raw_data_1Day_*_site_253_*.csv")
    files = sorted(glob.glob(pattern))
    # Prefer non-(1) 2019 file to avoid duplicates
    to_load = []
    seen_year = set()
    for f in files:
        basename = os.path.basename(f)
        if "(1)" in basename:
            continue  # skip duplicate 2019
        if "2017" in basename:
            y = 2017
        elif "2018" in basename:
            y = 2018
        elif "2019" in basename:
            y = 2019
        else:
            continue
        if y in seen_year:
            continue
        seen_year.add(y)
        to_load.append(f)

    dfs = []
    for f in to_load:
        df = pd.read_csv(f)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df["date"] = df["Timestamp"].dt.date
        # Replace "NA" string with NaN
        df = df.replace("NA", np.nan)
        for c in df.select_dtypes(include=["object"]).columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # Deduplicate by date (in case of overlaps)
    out = out.drop_duplicates(subset=["date"], keep="first")
    return out


def load_aqi(raw_dir: str) -> pd.DataFrame:
    """Load Ludhiana AQI bulletin, filter Ludhiana."""
    path = os.path.join(raw_dir, "Ludhiana_AQIBulletins (1).csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    if "City" in df.columns:
        df = df[df["City"].astype(str).str.upper().str.contains("LUDHIANA", na=False)]
    return df[["date", "Index Value", "Air Quality"]].copy()


def load_temperature(raw_dir: str) -> pd.DataFrame:
    """Load temperature_data, skip metadata rows."""
    path = os.path.join(raw_dir, "temperature_data.csv")
    df = pd.read_csv(path, skiprows=3)
    df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    return df


def merge_all(
    admissions: pd.DataFrame,
    pollutants: pd.DataFrame,
    aqi: pd.DataFrame,
    temperature: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all sources on date. Base = admission dates in 2017-2019 (maximize target coverage)."""
    # Base: admission dates in 2017-2019 (maximize target coverage)
    base = admissions.copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base[
        (base["date"].dt.year >= 2017) &
        (base["date"].dt.year <= 2019)
    ].drop_duplicates(subset=["date"])
    merged = base.copy()

    # Merge pollutants
    pol_cols = ["date", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "Ozone (µg/m³)", "CO (mg/m³)", "RH (%)"]
    pol_avail = [c for c in pol_cols if c in pollutants.columns]
    if pol_avail:
        pol_sub = pollutants[pol_avail].copy()
        pol_sub["date"] = pd.to_datetime(pol_sub["date"])
        merged = merged.merge(pol_sub, on="date", how="left", suffixes=("", "_dup"))
        merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

    # Merge AQI
    aqi["date"] = pd.to_datetime(aqi["date"])
    merged = merged.merge(aqi[["date", "Index Value", "Air Quality"]], on="date", how="left")

    # Merge temperature
    temp_cols = ["date", "temperature_2m_max (°C)", "temperature_2m_min (°C)", "apparent_temperature_max (°C)"]
    temp_avail = [c for c in temp_cols if c in temperature.columns]
    if temp_avail:
        temp_sub = temperature[["date"] + [c for c in temp_avail if c != "date"]].copy()
        temp_sub["date"] = pd.to_datetime(temp_sub["date"])
        merged = merged.merge(temp_sub, on="date", how="left")

    # Sort by date for proper time-ordered imputation (no future leakage)
    merged = merged.sort_values("date").reset_index(drop=True)

    # Drop rows outside pollutant date range (no PM data = would need future bfill = leakage)
    if not pollutants.empty:
        pol_first = pd.to_datetime(pollutants["date"]).min()
        pol_last = pd.to_datetime(pollutants["date"]).max()
        merged = merged[(merged["date"] >= pol_first) & (merged["date"] <= pol_last)]

    # Impute sparse features (AQI, temp gaps): ffill then bfill
    for c in FEATURE_COLS:
        if c in merged.columns:
            merged[c] = merged[c].ffill().bfill()

    return merged


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    out_path = os.path.join(data_dir, "merged_data.csv")

    print("Loading raw datasets...")
    hdhi_path = os.path.join(raw_dir, "HDHI Admission data - HDHI Admission data.csv")
    hdhi = load_hdhi_admissions(hdhi_path)
    admissions = aggregate_admissions(hdhi)
    print(f"  HDHI: {len(admissions)} unique admission dates")

    pollutants = load_raw_pollutants(raw_dir)
    print(f"  Raw pollutants: {len(pollutants)} rows, {pollutants['date'].min()} to {pollutants['date'].max()}")

    aqi = load_aqi(raw_dir)
    print(f"  AQI: {len(aqi)} rows")

    temperature = load_temperature(raw_dir)
    print(f"  Temperature: {len(temperature)} rows")

    print("\nMerging...")
    merged = merge_all(admissions, pollutants, aqi, temperature)

    # Build output with Timestamp and required columns
    merged = merged.sort_values("date").reset_index(drop=True)
    merged[TIMESTAMP_COL] = merged["date"].dt.strftime("%Y-%m-%d")

    out_cols = [TIMESTAMP_COL]
    for c in FEATURE_COLS:
        if c in merged.columns:
            out_cols.append(c)
    if "Air Quality" in merged.columns:
        out_cols.append("Air Quality")
    out_cols.extend([TARGET_COL])
    if "Brought_Dead_Count" in merged.columns:
        out_cols.append("Brought_Dead_Count")

    # Include extra pollutant columns for compatibility with original merged_data
    extra = [c for c in merged.columns if c not in out_cols and c not in ("date",)]
    out_cols = [c for c in out_cols if c in merged.columns] + [x for x in extra if x in merged.columns]

    merged_out = merged[[c for c in out_cols if c in merged.columns]].copy()
    merged_out.to_csv(out_path, index=False)
    ts = merged_out[TIMESTAMP_COL]
    print(f"\nWrote {out_path}: {len(merged_out)} rows, {ts.min()} to {ts.max()}")
    print(f"Columns: {list(merged_out.columns)}")


if __name__ == "__main__":
    main()

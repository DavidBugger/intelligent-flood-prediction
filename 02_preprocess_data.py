"""
=============================================================
INTELLIGENT FLOOD PREDICTION - NIGERIAN COASTAL CITIES
Step 2: Data Preprocessing & Merging
=============================================================
This script:
 - Cleans each dataset individually
 - Merges everything into one master training DataFrame
 - Engineers flood-relevant features
 - Saves the final dataset ready for ML modelling

Run this AFTER you have completed all downloads in 01_download_data.py
=============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("flood_project_data")
RAW      = BASE_DIR / "raw"
PROC     = BASE_DIR / "processed"


# ══════════════════════════════════════════════════════════
# SECTION 1 — LOAD & CLEAN RAINFALL DATA
# ══════════════════════════════════════════════════════════

def load_lagos_precipitation():
    """
    Loads Kaggle Lagos daily precipitation (2000-2023).
    Expected columns: date, rainfall_mm  (may vary — adjust below)
    """
    path = RAW / "rainfall/lagos_precipitation_2000_2023.csv"
    if not path.exists():
        print(" Lagos precipitation file not found. Skipping.")
        return None

    df = pd.read_csv(path)

    # ── Standardise column names (edit these to match your actual file) ──
    # Common column names from this Kaggle dataset:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check if date is in "days since 1970-01-01" format
    if "date_units" in df.columns and "days since 1970-01-01" in str(df["date_units"].iloc[0]):
        print("   Converting dates from 'days since 1970-01-01' format")
        df["date"] = pd.to_datetime(df["date"], unit="D", origin="1970-01-01")
    else:
        df["date"] = pd.to_datetime(df["date"])

    # Rename to standard names
    rename_map = {
        "time": "date",
        "precipitation": "rainfall_mm",
        "precip": "rainfall_mm",
        "rain": "rainfall_mm",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    df = df[["date", "rainfall_mm"]].dropna()
    df = df.sort_values("date").reset_index(drop=True)

    print(f" Lagos precipitation loaded: {len(df)} rows | {df['date'].min()} to {df['date'].max()}")
    return df


def load_hdx_subnational_rainfall():
    """
    Loads HDX CHIRPS subnational rainfall.
    Filters for Lagos and Port Harcourt states.
    """
    path = RAW / "rainfall/hdx_nigeria_rainfall_subnational_full.csv"
    if not path.exists():
        print("⚠️  HDX rainfall file not found. Skipping.")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Filter for relevant states (edit state names as they appear in your file)
    target_states = ["lagos", "rivers", "bayelsa", "delta", "akwa_ibom"]
    if "admin1name" in df.columns:
        df = df[df["admin1name"].str.lower().isin(target_states)]
    elif "state" in df.columns:
        df = df[df["state"].str.lower().isin(target_states)]

    # Parse date — HDX dekadal format is often like "2020-01-11"
    date_col = next((c for c in df.columns if "date" in c or "period" in c), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    # Keep just date, state, and rainfall value
    rain_col = next((c for c in df.columns if "rfh" in c or "rainfall" in c or "precip" in c), None)
    if rain_col:
        df = df[["date", rain_col]].rename(columns={rain_col: "regional_rainfall_mm"})
        df = df.groupby("date")["regional_rainfall_mm"].mean().reset_index()

    df = df.sort_values("date").reset_index(drop=True)
    print(f" HDX subnational rainfall loaded: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 2 — LOAD & CLEAN SEA LEVEL DATA
# ══════════════════════════════════════════════════════════

def load_psmsl_sea_level(city="lagos"):
    """
    Loads PSMSL monthly sea level data.
    PSMSL format: year;month;sea_level_mm;...
    """
    file_map = {
        "lagos": RAW / "sea_level/psmsl_lagos_monthly.csv",
        "port_harcourt": RAW / "sea_level/psmsl_port_harcourt_monthly.csv",
    }
    path = file_map[city]
    if not path.exists():
        print(f" PSMSL {city} file not found. Skipping.")
        return None

    # PSMSL files are semicolon-separated: Year Month SL-mm ...
    df = pd.read_csv(path, sep=";", header=None,
                     names=["year_month", "sea_level_mm", "flag1", "flag2"])

    # year_month format is e.g. "1953.0417" (year + fraction)
    df["year"] = df["year_month"].astype(str).str.split(".").str[0].astype(int)
    df["month_frac"] = df["year_month"] % 1
    df["month"] = (df["month_frac"] * 12 + 0.5).round().astype(int).clip(1, 12)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    df = df[["date", "sea_level_mm"]].dropna()
    df["sea_level_mm"] = pd.to_numeric(df["sea_level_mm"], errors="coerce")
    df = df[df["sea_level_mm"] != -99999]   # PSMSL missing value flag

    # Convert mm to metres
    df["sea_level_m"] = df["sea_level_mm"] / 1000
    df = df[["date", "sea_level_m"]].sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"sea_level_m": f"sea_level_m_{city}"})

    print(f" PSMSL {city} sea level loaded: {len(df)} rows")
    return df


def load_uhslc_tide():
    """
    Loads UHSLC daily tide data for Lagos.
    Format varies — adjust column names to match your downloaded file.
    """
    path = RAW / "sea_level/uhslc_lagos_daily.csv"
    if not path.exists():
        print(" UHSLC tide file not found. Skipping.")
        return None

    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Try to find date and sea level columns
    date_col = next((c for c in df.columns if "date" in c or "time" in c), None)
    sl_col   = next((c for c in df.columns if "sl" in c or "level" in c or "height" in c), None)

    if date_col and sl_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["tide_level_m"] = pd.to_numeric(df[sl_col], errors="coerce") / 1000
        df = df[["date", "tide_level_m"]].dropna()
    else:
        print(" Could not detect columns in UHSLC file. Check the file format.")
        return None

    df = df.sort_values("date").reset_index(drop=True)
    print(f" UHSLC tide data loaded: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 3 — LOAD FLOOD EVENT LABELS
# ══════════════════════════════════════════════════════════

def load_dartmouth_floods():
    """
    Loads Dartmouth Flood Observatory data.
    Filters for Nigeria and creates binary flood labels by date.
    """
    # Try both CSV and Excel versions
    csv_path = RAW / "flood_events/dartmouth_flood_archive.csv"
    xlsx_path = RAW / "flood_events/floodarchive.xlsx"
    
    if xlsx_path.exists():
        print(" Loading Dartmouth floods from Excel file...")
        df = pd.read_excel(xlsx_path)
    elif csv_path.exists():
        print(" Loading Dartmouth floods from CSV file...")
        df = pd.read_csv(csv_path, encoding="latin1")
    else:
        print(" Dartmouth flood archive not found. Skipping.")
        return None

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(f"   Original columns: {list(df.columns)}")

    # Filter for Nigeria - try different column name patterns
    country_col = None
    for col in df.columns:
        if "country" in col.lower():
            country_col = col
            break
    
    if country_col:
        nigeria_mask = df[country_col].astype(str).str.lower().str.contains("nigeria", na=False)
        df = df[nigeria_mask]
        print(f"   Nigeria events found: {len(df)}")
    else:
        print("   Could not find country column in Dartmouth file.")
        print("   Available columns:", list(df.columns))
        return None

    # Parse start and end dates - try different column name patterns
    begin_col = None
    end_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ["began", "start", "begin"]):
            begin_col = col
        if any(word in col_lower for word in ["ended", "end"]):
            end_col = col

    if not begin_col or not end_col:
        print("    Could not find date columns in Dartmouth file.")
        print(f"   Begin column candidates: {[c for c in df.columns if any(word in c.lower() for word in ['began', 'start', 'begin'])]}")
        print(f"   End column candidates: {[c for c in df.columns if any(word in c.lower() for word in ['ended', 'end'])]}")
        return None

    print(f"   Using date columns: {begin_col} (start), {end_col} (end)")
    
    df["flood_start"] = pd.to_datetime(df[begin_col], errors="coerce")
    df["flood_end"]   = pd.to_datetime(df[end_col],   errors="coerce")
    df = df.dropna(subset=["flood_start", "flood_end"])

    # Expand each flood event into a full date range
    all_flood_dates = set()
    for _, row in df.iterrows():
        dates = pd.date_range(row["flood_start"], row["flood_end"], freq="D")
        all_flood_dates.update(dates)

    flood_df = pd.DataFrame({"date": sorted(all_flood_dates)})
    flood_df["flood_occurred"] = 1
    print(f" Dartmouth floods loaded: {len(df)} Nigeria events → {len(flood_df)} flood days")
    return flood_df


def load_emdat_floods():
    """
    Loads EM-DAT Nigeria flood disaster data.
    Creates binary flood labels by date from disaster records.
    """
    path = RAW / "flood_events/emdat_nigeria_floods.xlsx"
    if not path.exists():
        print(". EM-DAT Nigeria floods file not found. Skipping.")
        return None

    print(" Loading EM-DAT floods from Excel file...")
    df = pd.read_excel(path)
    original_columns = list(df.columns)
    print(f"   Original columns: {original_columns}")

    # Filter for Nigeria and Flood disasters
    if "Country" in df.columns:
        nigeria_mask = df["Country"].astype(str).str.contains("Nigeria", na=False)
        df = df[nigeria_mask]
        print(f"   Nigeria events found: {len(df)}")
    
    if "Disaster Type" in df.columns:
        flood_mask = df["Disaster Type"].astype(str).str.contains("Flood", na=False)
        df = df[flood_mask]
        print(f"   Flood disasters found: {len(df)}")
    else:
        print(" Could not find 'Disaster Type' column in EM-DAT file.")
        return None

    # Create date from Start Year, Month, Day columns
    if all(col in df.columns for col in ["Start Year", "Start Month", "Start Day"]):
        # Convert day to integer (handle float values)
        df["start_day_int"] = df["Start Day"].fillna(1).astype(int)
        df["flood_start"] = pd.to_datetime(
            df["Start Year"].astype(str) + "-" + 
            df["Start Month"].astype(str).str.zfill(2) + "-" + 
            df["start_day_int"].astype(str).str.zfill(2),
            errors="coerce"
        )
        print(f"   Using Start Year/Month/Day columns for dates")
        print(f"   Date parsing success rate: {(~df['flood_start'].isna()).mean()*100:.1f}%")
    else:
        print("⚠️  Could not find Start Year/Month/Day columns in EM-DAT file.")
        print(f"   Available date columns: {[c for c in df.columns if any(word in str(c).lower() for word in ['year', 'month', 'day', 'date'])]}")
        return None
    
    df = df.dropna(subset=["flood_start"])

    # For EM-DAT, we'll mark the start date as flood day (could be extended later)
    flood_df = df[["flood_start"]].copy().rename(columns={"flood_start": "date"})
    flood_df["flood_occurred"] = 1
    flood_df = flood_df.drop_duplicates().sort_values("date").reset_index(drop=True)

    print(f" EM-DAT floods loaded: {len(df)} disasters → {len(flood_df)} flood days")
    print(f"   Date range: {flood_df['date'].min()} to {flood_df['date'].max()}")
    return flood_df


# ══════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Creates ML-ready features from the merged daily dataset.
    """
    df = df.sort_values("date").reset_index(drop=True)

    if "rainfall_mm" in df.columns:
        # Rolling rainfall accumulations
        df["rain_3day"]  = df["rainfall_mm"].rolling(3,  min_periods=1).sum()
        df["rain_7day"]  = df["rainfall_mm"].rolling(7,  min_periods=1).sum()
        df["rain_14day"] = df["rainfall_mm"].rolling(14, min_periods=1).sum()
        df["rain_30day"] = df["rainfall_mm"].rolling(30, min_periods=1).sum()

        # Rainfall intensity flag
        df["heavy_rain_flag"] = (df["rainfall_mm"] > 50).astype(int)

    if "tide_level_m" in df.columns:
        # Rolling tide stats
        df["tide_7day_mean"] = df["tide_level_m"].rolling(7, min_periods=1).mean()
        df["tide_7day_max"]  = df["tide_level_m"].rolling(7, min_periods=1).max()

        # High tide flag (above 75th percentile)
        threshold = df["tide_level_m"].quantile(0.75)
        df["high_tide_flag"] = (df["tide_level_m"] > threshold).astype(int)

    # Combined risk: heavy rain + high tide at the same time
    if "heavy_rain_flag" in df.columns and "high_tide_flag" in df.columns:
        df["combined_risk_flag"] = (
            (df["heavy_rain_flag"] == 1) & (df["high_tide_flag"] == 1)
        ).astype(int)

    # Calendar features
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear

    # Wet season flag (Nigeria wet season: April–October)
    df["wet_season"] = df["month"].between(4, 10).astype(int)

    print("✅ Feature engineering complete!")
    print(f"   Features created: {[c for c in df.columns if c != 'date']}")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 5 — MERGE ALL DATASETS
# ══════════════════════════════════════════════════════════

def build_master_dataset():
    print("\n" + "=" * 60)
    print("🔧 BUILDING MASTER DATASET")
    print("=" * 60)


    # ── Load all datasets ────────────────────────────────
    lagos_rain    = load_lagos_precipitation()
    regional_rain = load_hdx_subnational_rainfall()
    sea_lvl_lagos = load_psmsl_sea_level("lagos")
    sea_lvl_ph    = load_psmsl_sea_level("port_harcourt")
    tide_daily    = load_uhslc_tide()
    dartmouth_floods = load_dartmouth_floods()
    emdat_floods = load_emdat_floods()

    # ── Use Lagos rain as the date backbone ──────────────
    # Fall back to HDX if Kaggle file is missing
    if lagos_rain is not None:
        master = lagos_rain.copy()
    elif regional_rain is not None:
        master = pd.DataFrame({"date": regional_rain["date"]})
        master["rainfall_mm"] = np.nan
    else:
        print("❌ No rainfall data found. Cannot build master dataset.")
        return None

    # ── Upsample monthly sea level to daily (forward fill) ─
    if sea_lvl_lagos is not None:
        # Monthly → Daily
        sea_lvl_lagos = sea_lvl_lagos.set_index("date").resample("D").ffill().reset_index()
        master = master.merge(sea_lvl_lagos, on="date", how="left")

    if sea_lvl_ph is not None:
        sea_lvl_ph = sea_lvl_ph.set_index("date").resample("D").ffill().reset_index()
        master = master.merge(sea_lvl_ph, on="date", how="left")

    # ── Merge daily tide data ────────────────────────────
    if tide_daily is not None:
        master = master.merge(tide_daily, on="date", how="left")

    # ── Merge regional rainfall ──────────────────────────
    if regional_rain is not None:
        # HDX is dekadal (every 10 days) — resample to daily
        regional_rain = regional_rain.set_index("date").resample("D").interpolate().reset_index()
        master = master.merge(regional_rain, on="date", how="left")

    # ── Merge flood labels ───────────────────────────────
    flood_labels = None
    
    # Combine both flood data sources if available
    if dartmouth_floods is not None and emdat_floods is not None:
        # Combine both datasets
        combined_floods = pd.concat([dartmouth_floods, emdat_floods], ignore_index=True)
        # Remove duplicates and sort
        flood_labels = combined_floods.drop_duplicates(subset=["date"]).sort_values("date")
        print(f"✅ Combined flood labels: {len(dartmouth_floods)} Dartmouth + {len(emdat_floods)} EM-DAT → {len(flood_labels)} unique flood days")
    elif dartmouth_floods is not None:
        flood_labels = dartmouth_floods
        print(f"✅ Using Dartmouth flood labels only: {len(flood_labels)} flood days")
    elif emdat_floods is not None:
        flood_labels = emdat_floods
        print(f"✅ Using EM-DAT flood labels only: {len(flood_labels)} flood days")
    else:
        print("⚠️  No flood labels available yet. 'flood_occurred' set to NaN.")
        print("    Add Dartmouth/EM-DAT data and re-run to get labels.")

    if flood_labels is not None:
        master = master.merge(flood_labels, on="date", how="left")
        master["flood_occurred"] = master["flood_occurred"].fillna(0).astype(int)
    else:
        # No labels yet — add placeholder column
        master["flood_occurred"] = np.nan

    # ── Feature engineering ──────────────────────────────
    master = engineer_features(master)

    # ── Final cleanup ────────────────────────────────────
    master = master.sort_values("date").reset_index(drop=True)

    # Summary
    print(f"\n📊 Master dataset shape: {master.shape}")
    print(f"   Date range : {master['date'].min()} to {master['date'].max()}")
    if "flood_occurred" in master.columns and not master["flood_occurred"].isna().all():
        flood_count = master["flood_occurred"].sum()
        print(f"   Flood days : {int(flood_count)} ({flood_count/len(master)*100:.1f}%)")
    print(f"   Columns    : {list(master.columns)}")

    # Save
    out_path = PROC / "master_dataset.csv"
    master.to_csv(out_path, index=False)
    print(f"\n✅ Master dataset saved to: {out_path}")

    return master


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    master = build_master_dataset()

    if master is not None:
        print("\n🔍 Preview of master dataset:")
        print(master.head(10).to_string())
        print("\n📈 Missing value summary:")
        print(master.isnull().sum())

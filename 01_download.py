"""
=============================================================
INTELLIGENT FLOOD PREDICTION - NIGERIAN COASTAL CITIES
Step 1: Data Download Script
=============================================================
This script downloads all required datasets automatically
where possible, and gives instructions for manual downloads.

Author: MSc AI Project
=============================================================
"""

import os
import requests
import pandas as pd
from pathlib import Path

# ── Create folder structure ───────────────────────────────
BASE_DIR = Path("flood_project_data")
FOLDERS = [
    "raw/rainfall",
    "raw/sea_level",
    "raw/geospatial",
    "raw/flood_events",
    "processed",
    "models",
    "outputs/maps",
    "outputs/plots",
]
for folder in FOLDERS:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

print(" Folder structure created!\n")
print(BASE_DIR)
for f in FOLDERS:
    print(f"  └── {f}")


# ──────────────────────────────────────────────────────────
# DATASET 1: World Bank Nigeria Monthly Precipitation
# (Auto-download — no login required)
# ──────────────────────────────────────────────────────────
def download_worldbank_precipitation():
    print("\n Downloading World Bank Nigeria Precipitation Data...")

    # ── New API structure (updated 2024) ──────────────────────────
    # Template: /cckp/v1/{params}/{geocode}?format=xlsx
    # Breaking down the parameter string:
    #   collection  : cru-x0.5
    #   type        : timeseries
    #   variable    : pr  (precipitation)
    #   product     : timeseries
    #   aggregation : monthly
    #   period      : 1901-2024
    #   percentile  : mean
    #   scenario    : historical
    #   model       : cru_ts4.09
    #   calculation : mean
    #   geocode     : NGA  ← now a SEPARATE path segment

    BASE     = "https://cckpapi.worldbank.org/cckp/v1"
    PARAMS   = "cru-x0.5_timeseries_pr_timeseries_monthly_1901-2024_mean_historical_cru_ts4.09_mean"
    GEOCODE  = "NGA"
    FORMAT   = "json"

    url = f"{BASE}/{PARAMS}/{GEOCODE}?format={FORMAT}"
    print(f"   Requesting: {url}")

    save_path = BASE_DIR / "raw/rainfall/worldbank_nigeria_precipitation_1901_2024.xlsx"

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        # Sanity check — make sure we actually got an Excel file
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise ValueError(
                f"Got HTML instead of XLSX — the URL may still be wrong.\n"
                f"Content-Type: {content_type}\n"
                f"Response preview: {r.text[:300]}"
            )

        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"   Saved to: {save_path} ({len(r.content)/1024:.1f} KB)")

    except ValueError as e:
        print(f"     Bad response: {e}")
        print(f"     Try opening this URL manually in your browser to verify:")
        print(f"      {url}")
    except Exception as e:
        print(f"     Request failed: {e}")
        print(f"     Manually download from: {url}")

# ──────────────────────────────────────────────────────────
# DATASET 2: HDX Nigeria Subnational Rainfall (Full)
# (Auto-download — no login required)
# ──────────────────────────────────────────────────────────
def download_hdx_rainfall_full():
    print("\n. Downloading HDX Nigeria Subnational Rainfall (Full)...")
    url = (
        "https://data.humdata.org/dataset/114874de-df99-4102-b4c8-b44e2db44a5e"
        "/resource/114874de-df99-4102-b4c8-b44e2db44a5e"
        "/download/nga-rainfall-subnat-full.csv"
    )
    save_path = BASE_DIR / "raw/rainfall/hdx_nigeria_rainfall_subnational_full.csv"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"     Saved to: {save_path}")
    except Exception as e:
        print(f"     Failed: {e}")
        print("      Manually download from:")
        print(f"      {url}")


# ──────────────────────────────────────────────────────────
# DATASET 3: HDX Nigeria Subnational Rainfall (5-Year)
# (Auto-download — no login required)
# ──────────────────────────────────────────────────────────
def download_hdx_rainfall_5ytd():
    print("\n Downloading HDX Nigeria Subnational Rainfall (5-Year)...")
    url = (
        "https://data.humdata.org/dataset/2670aa7f-21c1-4ebc-88d6-8d99683e5a60"
        "/resource/2670aa7f-21c1-4ebc-88d6-8d99683e5a60"
        "/download/nga-rainfall-subnat-5ytd.csv"
    )
    save_path = BASE_DIR / "raw/rainfall/hdx_nigeria_rainfall_subnational_5ytd.csv"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"     Saved to: {save_path}")
    except Exception as e:
        print(f"     Failed: {e}")
        print("      Manually download from:")
        print(f"      {url}")


# ──────────────────────────────────────────────────────────
# MANUAL DOWNLOAD INSTRUCTIONS
# (Datasets that require login or button clicks)
# ──────────────────────────────────────────────────────────
def print_manual_instructions():
    print("\n" + "=" * 65)
    print(" MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 65)

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 4 — Lagos Precipitation 2000-2023 (Kaggle)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Log in or create free account at: https://www.kaggle.com
2. Go to: https://www.kaggle.com/datasets/sodipepaul/lagos-precipitation-data-2000-2023
3. Click the Download button
4. Save the CSV to: flood_project_data/raw/rainfall/lagos_precipitation_2000_2023.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 5 — Sea Level Lagos Station 162 (PSMSL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://psmsl.org/data/obtaining/stations/162.php
2. Click "Download monthly mean sea level data."
3. Save to: flood_project_data/raw/sea_level/psmsl_lagos_monthly.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 6 — Sea Level Port Harcourt Station 404 (PSMSL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://psmsl.org/data/obtaining/stations/404.php
2. Click "Download monthly mean sea level data."
3. Save to: flood_project_data/raw/sea_level/psmsl_port_harcourt_monthly.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 7 — Lagos Tide Daily & Hourly (UHSLC Station 233)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://uhslc.soest.hawaii.edu/data/?rq#233
2. Click "daily" next to Lagos → save as:
   flood_project_data/raw/sea_level/uhslc_lagos_daily.csv
3. Click "hourly" next to Lagos → save as:
   flood_project_data/raw/sea_level/uhslc_lagos_hourly.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 8 — SRTM Digital Elevation Model (USGS EarthExplorer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Register free at: https://earthexplorer.usgs.gov/
2. Search Criteria → World Features → search "Nigeria"
3. Data Sets → Digital Elevation → SRTM → SRTM 1 Arc-Second Global
4. Results → Download
5. Save .tif file to: flood_project_data/raw/geospatial/srtm_nigeria.tif

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 9 — Historical Flood Events (Dartmouth Flood Observatory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://floodobservatory.colorado.edu/Archives/index.html
2. Download the global flood archive (Excel/CSV)
3. Save to: flood_project_data/raw/flood_events/dartmouth_flood_archive.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 10 — EM-DAT Disaster Database
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Register free at: https://www.emdat.be/
2. Search: Country = Nigeria, Disaster Type = Flood
3. Export results as Excel
4. Save to: flood_project_data/raw/flood_events/emdat_nigeria_floods.xlsx

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET 11 — Nigeria Admin Boundaries (GADM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://gadm.org/download_country.html
2. Select "Nigeria" and download "Shapefile"
3. Extract the .zip and save to:
   flood_project_data/raw/geospatial/gadm_nigeria/

""")
    print("=" * 65)
    print(" Once all files are downloaded, run: 02_preprocess_data.py")
    print("=" * 65)


# ── Run all downloads ─────────────────────────────────────
if __name__ == "__main__":
    download_worldbank_precipitation()
    download_hdx_rainfall_full()
    download_hdx_rainfall_5ytd()
    print_manual_instructions()
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load metadata to get feature columns
BASE_DIR = Path("/Users/mac/Desktop/PROJECTS/MSC")
MODELS_DIR = BASE_DIR / "/Users/mac/Desktop/PROJECTS/MSC/flood_project_data/models"

with open(MODELS_DIR / "model_meta.json", "r") as f:
    meta = json.load(f)
FEATURE_COLS = meta["feature_cols"]

def generate_high_risk_data():
    scenarios = []
    
    # 1. Extreme Flash Flood (Lagos)
    scenarios.append({
        "scenario_name": "Extreme Flash Flood - Lagos",
        "rainfall_mm": 185.0,
        "rain_3day": 250.0,
        "rain_7day": 320.0,
        "rain_14day": 400.0,
        "rain_30day": 600.0,
        "sea_level_m_lagos": 6.8,
        "sea_level_m_port_harcourt": 6.5,
        "month": 7,
        "day_of_year": 190,
        "heavy_rain_flag": 1,
        "wet_season": 1
    })

    # 2. Prolonged Seasonal Saturation (Month 8 - August)
    scenarios.append({
        "scenario_name": "Prolonged Cumulative Saturation",
        "rainfall_mm": 95.0,
        "rain_3day": 300.0,
        "rain_7day": 550.0,
        "rain_14day": 900.0,
        "rain_30day": 1800.0,
        "sea_level_m_lagos": 5.5,
        "sea_level_m_port_harcourt": 5.2,
        "month": 8,
        "day_of_year": 220,
        "heavy_rain_flag": 1,
        "wet_season": 1
    })

    # 3. Tidal Surge + Heavy Rain (Port Harcourt)
    scenarios.append({
        "scenario_name": "Tidal Surge + Heavy Rain",
        "rainfall_mm": 110.0,
        "rain_3day": 180.0,
        "rain_7day": 250.0,
        "rain_14day": 350.0,
        "rain_30day": 500.0,
        "sea_level_m_lagos": 7.5,
        "sea_level_m_port_harcourt": 7.8,
        "month": 6,
        "day_of_year": 165,
        "heavy_rain_flag": 1,
        "wet_season": 1
    })

    # 4. Late Season Extreme (October)
    scenarios.append({
        "scenario_name": "Late Season Storm",
        "rainfall_mm": 140.0,
        "rain_3day": 220.0,
        "rain_7day": 400.0,
        "rain_14day": 600.0,
        "rain_30day": 1100.0,
        "sea_level_m_lagos": 6.2,
        "sea_level_m_port_harcourt": 6.0,
        "month": 10,
        "day_of_year": 285,
        "heavy_rain_flag": 1,
        "wet_season": 1
    })
    
    # 5. Delta Region Emergency (Warri/PH)
    scenarios.append({
        "scenario_name": "Delta Region Risk Combo",
        "rainfall_mm": 160.0,
        "rain_3day": 350.0,
        "rain_7day": 600.0,
        "rain_14day": 1000.0,
        "rain_30day": 2200.0,
        "sea_level_m_lagos": 5.0,
        "sea_level_m_port_harcourt": 7.2,
        "month": 9,
        "day_of_year": 255,
        "heavy_rain_flag": 1,
        "wet_season": 1
    })

    df = pd.DataFrame(scenarios)
    
    # Ensure all required features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            
    # Reorder columns to match model expectations just in case
    cols = ["scenario_name"] + [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols]
    
    output_path = BASE_DIR / "flood_project_data/outputs/high_risk_test_scenarios.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} high-risk scenarios at: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_high_risk_data()

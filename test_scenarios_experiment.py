import pickle
import numpy as np
import json
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/mac/Desktop/PROJECTS/MSC")
MODELS_DIR = BASE_DIR / "flood_project_data" / "models"

# Load Model, Scaler, and Meta
with open(MODELS_DIR / "xgboost_flood_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(MODELS_DIR / "model_meta.json", "r") as f:
    meta = json.load(f)
FEATURE_COLS = meta["feature_cols"]

def test_custom(values):
    # Start with Row 3648 values as a base since we KNOW it works
    df = pd.read_csv(BASE_DIR / "flood_project_data/processed/master_dataset.csv")
    base_row = df.iloc[3648][FEATURE_COLS].to_dict()
    
    # Override with our test values
    base_row.update(values)
    
    X = pd.DataFrame([base_row])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    return base_row, prob

# Scenario A: Row 3648 with HIGH rainfall
vals_a = {
    "rainfall_mm": 150.0,
    "rain_7day": 400.0,
    "rain_30day": 1000.0,
    "heavy_rain_flag": 1
}

# Scenario B: High rainfall but WRONG month (January)
vals_b = {
    "rainfall_mm": 150.0,
    "rain_7day": 400.0,
    "rain_30day": 1000.0,
    "heavy_rain_flag": 1,
    "month": 1,
    "day_of_year": 15,
    "wet_season": 0
}

# Scenario C: Low rainfall but WRONG sea level
vals_c = {
    "sea_level_m_lagos": 1.5 # The value I was suggesting before!
}

row_a, prob_a = test_custom(vals_a)
row_b, prob_b = test_custom(vals_b)
row_c, prob_c = test_custom(vals_c)

print(f"High Rain + Sept: {prob_a:.4f}")
print(f"High Rain + Jan: {prob_b:.4f}")
print(f"Low Sea Level: {prob_c:.4f}")

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
    # Base: High-risk historical row
    df = pd.read_csv(BASE_DIR / "flood_project_data/processed/master_dataset.csv")
    base_row = df.iloc[3648][FEATURE_COLS].to_dict()
    base_row.update(values)
    
    X = pd.DataFrame([base_row])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    return base_row, prob

# Scenario D: The "True" High Risk
vals_d = {
    "rainfall_mm": 190.0,
    "rain_7day": 500.0,
    "rain_30day": 1200.0,
    "month": 7,       # July - Peak Risk
    "day_of_year": 200,
    "sea_level_m_lagos": 7.25,
    "heavy_rain_flag": 1
}

vals_e = {
    "rainfall_mm": 10.0,
    "rain_7day": 40.0,
    "month": 7,
    "day_of_year": 200,
    "sea_level_m_lagos": 7.09
}

row_d, prob_d = test_custom(vals_d)
row_e, prob_e = test_custom(vals_e)

print(f"Red Alert Attempt: {prob_d:.4f}")
print(f"Baseline (Low): {prob_e:.4f}")

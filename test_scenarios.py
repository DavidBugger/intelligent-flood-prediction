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

def test_row(row_idx):
    df = pd.read_csv(BASE_DIR / "flood_project_data/processed/master_dataset.csv")
    row = df.iloc[row_idx][FEATURE_COLS].to_dict()
    X = pd.DataFrame([row])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    return row, prob

row_vals, prob = test_row(3648)
print(f"Row 3648 Prob: {prob:.4f}")
print(f"Row Values: {row_vals}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="Nigerian Flood Early Warning API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "flood_project_data" / "models"
DATA_DIR = BASE_DIR / "flood_project_data" / "processed"
OUTPUTS_DIR = BASE_DIR / "flood_project_data" / "outputs"

# Load Model, Scaler, and Meta
try:
    with open(MODELS_DIR / "xgboost_flood_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "model_meta.json", "r") as f:
        meta = json.load(f)
    FEATURE_COLS = meta["feature_cols"]
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    FEATURE_COLS = []

class PredictionInput(BaseModel):
    rainfall_mm: float
    rain_7day: float
    rain_3day: Optional[float] = None
    rain_14day: Optional[float] = None
    rain_30day: Optional[float] = None
    heavy_rain_flag: Optional[int] = None
    tide_level_m: Optional[float] = None
    sea_level_m_lagos: Optional[float] = None
    sea_level_m_port_harcourt: Optional[float] = None
    month: int
    day_of_year: Optional[int] = None
    wet_season: Optional[int] = None

class PredictionOutput(BaseModel):
    probability: float
    alert_level: str
    emoji: str
    label: str
    action: str
    color: str

def get_alert_level(probability: float) -> dict:
    if probability < 0.35:
        return {
            "level": "GREEN",
            "emoji": "🟢",
            "label": "Low Risk",
            "action": "Normal operations. Monitor conditions.",
            "color": "#4CAF50",
        }
    elif probability < 0.65:
        return {
            "level": "AMBER",
            "emoji": "🟡",
            "label": "Moderate Risk",
            "action": "Alert local authorities. Prepare drainage. Issue public advisory.",
            "color": "#FF9800",
        }
    else:
        return {
            "level": "RED",
            "emoji": "🔴",
            "label": "HIGH RISK — Flood Warning",
            "action": "IMMEDIATE ACTION: Evacuate low-lying areas. Activate emergency response.",
            "color": "#F44336",
        }

@app.get("/")
async def root():
    return {"status": "online", "message": "Nigerian Flood Early Warning System API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded")

    # Prepare input data
    input_dict = {col: 0.0 for col in FEATURE_COLS}
    
    # Fill in provided values
    provided_data = data.dict()
    for key, value in provided_data.items():
        if key in FEATURE_COLS:
            input_dict[key] = value

    # Computed fields if not provided
    if "wet_season" in FEATURE_COLS and data.wet_season is None:
        input_dict["wet_season"] = 1 if 4 <= data.month <= 10 else 0
    
    if "heavy_rain_flag" in FEATURE_COLS and data.heavy_rain_flag is None:
        input_dict["heavy_rain_flag"] = 1 if data.rainfall_mm > 50 else 0
        
    if "day_of_year" in FEATURE_COLS and data.day_of_year is None:
        # Match day_of_year to the selected month instead of today's date
        input_dict["day_of_year"] = data.month * 30 - 15

    # SMART DEFAULTS for missing rain lags
    if "rain_3day" in FEATURE_COLS and (data.rain_3day is None or data.rain_3day == 0):
        input_dict["rain_3day"] = data.rain_7day * 0.45
    if "rain_14day" in FEATURE_COLS and (data.rain_14day is None or data.rain_14day == 0):
        input_dict["rain_14day"] = data.rain_7day * 1.8
    if "rain_30day" in FEATURE_COLS and (data.rain_30day is None or data.rain_30day == 0):
        input_dict["rain_30day"] = data.rain_7day * 3.5

    # Sea Level Consistency (Must be ~7.0 in this dataset)
    if "sea_level_m_lagos" in FEATURE_COLS and data.sea_level_m_lagos is None:
        input_dict["sea_level_m_lagos"] = 7.07
        
    if "sea_level_m_port_harcourt" in FEATURE_COLS:
        ph_val = data.sea_level_m_port_harcourt
        if ph_val is None or ph_val < 1:
             # Port Harcourt must also follow the ~7.0m range of the training data
             input_dict["sea_level_m_port_harcourt"] = input_dict.get("sea_level_m_lagos", 7.07) - 0.02
        else:
             input_dict["sea_level_m_port_harcourt"] = ph_val

    # Convert to array and scale
    X_raw = np.array([[input_dict[c] for c in FEATURE_COLS]])
    
    # DEBUG LOGS
    print("\n" + "="*50)
    print("      FLOOD RISK PREDICTION DEBUG")
    print("="*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Payload Month: {data.month}")
    print("\nFeatures passed to model:")
    for c in FEATURE_COLS:
        print(f"  - {c:30}: {input_dict[c]}")
    
    X_scaled = scaler.transform(X_raw)
    print("\nScaled values (first 5):", X_scaled[0][:5])
    
    # Predict
    prob = float(model.predict_proba(X_scaled)[0][1])
    alert = get_alert_level(prob)
    
    print(f"\nRESULT -> Probability: {prob:.4%}")
    print(f"RESULT -> Alert Level: {alert['level']}")
    print("="*50 + "\n")
    
    return {
        "probability": prob,
        "alert_level": alert["level"],
        "emoji": alert["emoji"],
        "label": alert["label"],
        "action": alert["action"],
        "color": alert["color"]
    }

@app.get("/history")
async def history(limit: int = 30):
    try:
        csv_path = OUTPUTS_DIR / "flood_predictions.csv"
        if not csv_path.exists():
            return []
        
        df = pd.read_csv(csv_path)
        # Take the last 'limit' items
        recent = df.tail(limit).to_dict(orient="records")
        return recent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

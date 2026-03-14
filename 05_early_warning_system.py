"""
=============================================================
INTELLIGENT FLOOD PREDICTION - NIGERIAN COASTAL CITIES
Step 5: Early Warning System
=============================================================
Takes the trained XGBoost model and generates:
  - Flood risk predictions for new/recent data
  - Green / Amber / Red alert levels
  - A simple alert report output
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("flood_project_data")
PROC     = BASE_DIR / "processed"
MODELS   = BASE_DIR / "models"
OUTPUTS  = BASE_DIR / "outputs"


# ──────────────────────────────────────────────────────────
# LOAD MODEL AND SCALER
# ──────────────────────────────────────────────────────────
with open(MODELS / "xgboost_flood_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODELS / "model_meta.json") as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
print(f" Model loaded. Using {len(FEATURE_COLS)} features.")


# ──────────────────────────────────────────────────────────
# ALERT LEVEL FUNCTION
# ──────────────────────────────────────────────────────────
def get_alert_level(probability: float) -> dict:
    """
    Converts flood probability into a 3-tier alert system.

    Returns a dict with:
      - level: GREEN / AMBER / RED
      - emoji: visual indicator
      - action: recommended action
    """
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
            "action": "Alert local authorities. Prepare drainage. "
                      "Issue public advisory.",
            "color": "#FF9800",
        }
    else:
        return {
            "level": "RED",
            "emoji": "🔴",
            "label": "HIGH RISK — Flood Warning",
            "action": "IMMEDIATE ACTION: Evacuate low-lying areas. "
                      "Activate emergency response. Close flood barriers.",
            "color": "#F44336",
        }


# ──────────────────────────────────────────────────────────
# PREDICT ON FULL DATASET
# ──────────────────────────────────────────────────────────
def generate_predictions():
    df = pd.read_csv(PROC / "master_dataset.csv", parse_dates=["date"])

    # Only keep rows with all required features
    df_feat = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    X_scaled = scaler.transform(df_feat)

    probs = model.predict_proba(X_scaled)[:, 1]

    df["flood_probability"]  = probs
    df["predicted_flood"]    = (probs >= 0.5).astype(int)
    df["alert_level"]        = df["flood_probability"].apply(
        lambda p: get_alert_level(p)["level"]
    )

    # Save predictions
    out = df[["date", "rainfall_mm", "flood_probability",
              "predicted_flood", "alert_level"]].copy()
    if "flood_occurred" in df.columns:
        out["actual_flood"] = df["flood_occurred"]

    out.to_csv(OUTPUTS / "flood_predictions.csv", index=False)
    print(f"✅ Predictions saved to: {OUTPUTS}/flood_predictions.csv")
    return out


# ──────────────────────────────────────────────────────────
# ALERT REPORT FUNCTION
# ──────────────────────────────────────────────────────────
def generate_alert_report(predictions_df):
    """
    Generates a text-based early warning report for the most recent 30 days.
    In a real system this would be emailed or pushed to a dashboard.
    """
    recent = predictions_df.tail(30).copy()
    today_row = recent.iloc[-1]
    today_alert = get_alert_level(today_row["flood_probability"])

    report = f"""
╔══════════════════════════════════════════════════════╗
║   NIGERIAN COASTAL FLOOD EARLY WARNING SYSTEM        ║
║   Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}            ║
╚══════════════════════════════════════════════════════╝

TODAY'S FORECAST
────────────────────────────────────────────────────
Date              : {today_row['date'].strftime('%Y-%m-%d')}
Flood Probability : {today_row['flood_probability']:.1%}
Alert Level       : {today_alert['emoji']}  {today_alert['label']}
Recommended Action: {today_alert['action']}

LAST 30 DAYS SUMMARY
────────────────────────────────────────────────────
Red   (High Risk)   days: {(recent['alert_level'] == 'RED').sum()}
Amber (Medium Risk) days: {(recent['alert_level'] == 'AMBER').sum()}
Green (Low Risk)    days: {(recent['alert_level'] == 'GREEN').sum()}
Average Flood Prob        : {recent['flood_probability'].mean():.1%}
Peak Flood Probability    : {recent['flood_probability'].max():.1%}
    on {recent.loc[recent['flood_probability'].idxmax(), 'date'].strftime('%Y-%m-%d')}

HIGH RISK DAYS IN LAST 30 DAYS
────────────────────────────────────────────────────"""

    high_risk = recent[recent["alert_level"] == "RED"]
    if len(high_risk) == 0:
        report += "\n  None — conditions appear stable."
    else:
        for _, row in high_risk.iterrows():
            report += f"\n  🔴 {row['date'].strftime('%Y-%m-%d')}  —  {row['flood_probability']:.1%} probability"

    report += """

CITIES MONITORED
────────────────────────────────────────────────────
  📍 Lagos (Primary)
  📍 Port Harcourt
  📍 Warri
  📍 Calabar

NOTE: This is an AI-generated forecast. Always verify
with local meteorological and hydrological authorities.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(report)

    # Save report
    report_path = OUTPUTS / "early_warning_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Alert report saved to: {report_path}")


# ──────────────────────────────────────────────────────────
# PREDICT ON NEW / LIVE DATA (Example Function)
# ──────────────────────────────────────────────────────────
def predict_single_day(
    rainfall_mm: float,
    rain_7day: float,
    tide_level_m: float,
    month: int,
    **kwargs
):
    """
    Predict flood risk for a single day given key inputs.

    Usage example:
        result = predict_single_day(
            rainfall_mm=85.0,
            rain_7day=210.0,
            tide_level_m=1.35,
            month=8
        )
    """
    row = {col: 0 for col in FEATURE_COLS}
    row.update({
        "rainfall_mm":    rainfall_mm,
        "rain_7day":      rain_7day,
        "tide_level_m":   tide_level_m,
        "month":          month,
        "wet_season":     1 if 4 <= month <= 10 else 0,
        "heavy_rain_flag": 1 if rainfall_mm > 50 else 0,
    })
    row.update(kwargs)

    X = np.array([[row[c] for c in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    alert = get_alert_level(prob)

    print(f"\n🌊 FLOOD RISK PREDICTION")
    print(f"   Rainfall Today : {rainfall_mm} mm")
    print(f"   7-Day Rain     : {rain_7day} mm")
    print(f"   Tide Level     : {tide_level_m} m")
    print(f"   Month          : {month}")
    print(f"   ────────────────────────────")
    print(f"   Flood Probability : {prob:.1%}")
    print(f"   Alert Level       : {alert['emoji']} {alert['label']}")
    print(f"   Action            : {alert['action']}")

    return {"probability": prob, "alert": alert}


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate full predictions
    preds = generate_predictions()

    # Generate alert report for most recent period
    generate_alert_report(preds)

    # Example: predict for a hypothetical rainy day
    print("\n" + "=" * 55)
    print("📡 EXAMPLE: Predicting a single day")
    print("=" * 55)
    predict_single_day(
        rainfall_mm=95.0,    # very heavy rain
        rain_7day=280.0,     # wet week
        tide_level_m=1.4,    # high tide
        month=8              # August — peak wet season
    )

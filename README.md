# 🌊 INTELLIGENT FLOOD PREDICTION — NIGERIAN COASTAL CITIES
## MSc AI Project Pipeline & Dashboard

---

## 📁 Project Structure

```
flood_project_data/
├── raw/                 ← Raw datasets (precipitation, sea level, etc.)
├── processed/           ← Merged training datasets
├── models/              ← Saved ML models (XGBoost, LSTM)
└── outputs/             ← EDA plots, maps, logs

dashboard/
├── backend/             ← FastAPI backend serving ML models
└── frontend/            ← Next.js interactive web dashboard

01_download.py           ← Data download script
02_preprocess_data.py    ← Data cleaning & merging
03_eda_analysis.py       ← Exploratory Data Analysis
04_train_model.py        ← Model training script
05_early_warning_system.py ← Generates initial reports
```

---

## 🚀 How to Run the ML Pipeline (Step by Step)

### Step 0 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 1 — Download Data
```bash
python 01_download.py
```
Follow the printed manual download instructions for Kaggle, PSMSL, UHSLC, USGS, Dartmouth, EM-DAT and GADM.

### Step 2 — Preprocess & Merge
```bash
python 02_preprocess_data.py
```

### Step 3 — Exploratory Data Analysis
```bash
python 03_eda_analysis.py
```

### Step 4 — Train Models
```bash
python 04_train_model.py
```

### Step 5 — Early Warning System
```bash
python 05_early_warning_system.py
```

---

## 🌐 How to Run the Web Dashboard

### 1. Start the Backend (FastAPI)
The backend serves the trained ML models to the frontend.
```bash
cd dashboard/backend
pip install -r requirements.txt
python main.py
```
*The API will be available at `http://localhost:8000`*

### 2. Start the Frontend (Next.js)
The frontend provides an interactive UI for the flood Early Warning System.
```bash
cd dashboard/frontend
npm install
npm run dev
```
*The web dashboard will be available at `http://localhost:3000`*

---

## 📊 Data Sources Summary

| # | Dataset | Source | Format | Auto-Download? |
|---|---------|--------|--------|----------------|
| 1 | Lagos Precipitation 2000–2023 | Kaggle | CSV | Manual |
| 2 | Nigeria Monthly Precip 1901–2024 | World Bank | XLSX | ✅ Auto |
| 3 | Nigeria Subnational Rainfall | HDX/CHIRPS | CSV | ✅ Auto |
| 4 | Sea Level Lagos (Stn 162) | PSMSL | CSV | Manual |
| 5 | Sea Level Port Harcourt (Stn 404) | PSMSL | CSV | Manual |
| 6 | Tide Data Lagos (Stn 233) | UHSLC | CSV | Manual |
| 7 | SRTM DEM Nigeria | USGS EarthExplorer | GeoTIFF | Manual |
| 8 | Historical Flood Events | Dartmouth | CSV | Manual |
| 9 | Nigeria Disaster Records | EM-DAT | XLSX | Manual |
| 10| Nigeria Admin Boundaries | GADM | Shapefile | Manual |

---

## 🧠 Models

### Model A — XGBoost
- Fast, interpretable, great baseline.
- Works on tabular daily features.
- Output: flood probability (0–1) + alert level.

### Model B — LSTM
- Captures temporal patterns over a 14-day lookback window.
- Better suited for time series flood dynamics.
- Output: flood probability (0–1).

---

## 🚨 Alert Level System

| Level | Probability | Action |
|-------|------------|--------|
| 🟢 GREEN | < 35% | Monitor conditions |
| 🟡 AMBER | 35–65% | Alert authorities, prepare drainage |
| 🔴 RED | > 65% | Evacuate low-lying areas, emergency response |

---

## ⚠️ Important Notes

1. **Class imbalance**: Flood days are rare. Both models use class weighting.
2. **No data leakage**: Train/test split is chronological (first 80% train).
3. **Ground truth**: Download Dartmouth/EM-DAT to train a supervised model.
4. **Elevation data**: DEM is used in GIS (QGIS) and mapped separately.

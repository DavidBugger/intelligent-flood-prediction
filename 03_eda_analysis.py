"""
=============================================================
INTELLIGENT FLOOD PREDICTION - NIGERIAN COASTAL CITIES
Step 3: Exploratory Data Analysis (EDA)
=============================================================
Run this after 02_preprocess_data.py to visualise your data
before building any models.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("flood_project_data")
PROC     = BASE_DIR / "processed"
PLOTS    = BASE_DIR / "outputs/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Load master dataset ───────────────────────────────────
df = pd.read_csv(PROC / "master_dataset.csv", parse_dates=["date"])
print(f" Loaded master dataset: {df.shape[0]} rows, {df.shape[1]} columns")


# ══════════════════════════════════════════════════════════
# PLOT 1 — Annual Rainfall Trend
# ══════════════════════════════════════════════════════════
def plot_annual_rainfall():
    if "rainfall_mm" not in df.columns:
        return

    annual = df.groupby("year")["rainfall_mm"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(annual["year"], annual["rainfall_mm"], color="#2196F3", alpha=0.8)
    ax.set_title("Annual Rainfall — Lagos (mm)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Rainfall (mm)")

    # Add trend line
    z = np.polyfit(annual["year"], annual["rainfall_mm"], 1)
    p = np.poly1d(z)
    ax.plot(annual["year"], p(annual["year"]), "r--", linewidth=2, label="Trend")
    ax.legend()

    plt.tight_layout()
    path = PLOTS / "01_annual_rainfall_trend.png"
    plt.savefig(path, dpi=150)
    print(f" Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# PLOT 2 — Monthly Rainfall Seasonality
# ══════════════════════════════════════════════════════════
def plot_monthly_seasonality():
    if "rainfall_mm" not in df.columns:
        return

    monthly = df.groupby("month")["rainfall_mm"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(month_names, monthly["rainfall_mm"],
                  color=["#64B5F6" if m not in [4,5,6,7,8,9,10] else "#1565C0"
                         for m in monthly["month"]], alpha=0.9)
    ax.set_title("Average Monthly Rainfall — Lagos\n(Blue = Wet Season)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rainfall (mm)")
    ax.axvspan(3, 9, alpha=0.05, color="blue", label="Wet Season (Apr–Oct)")
    ax.legend()
    plt.tight_layout()
    path = PLOTS / "02_monthly_seasonality.png"
    plt.savefig(path, dpi=150)
    print(f" Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# PLOT 3 — Sea Level Trend Over Time
# ══════════════════════════════════════════════════════════
def plot_sea_level_trend():
    sl_col = next((c for c in df.columns if "sea_level" in c), None)
    if not sl_col:
        print("⚠️  No sea level column found — skipping plot 3")
        return

    annual_sl = df.groupby("year")[sl_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(annual_sl["year"], annual_sl[sl_col], color="#00897B", linewidth=2.5)
    ax.fill_between(annual_sl["year"], annual_sl[sl_col],
                    alpha=0.2, color="#00897B")
    ax.set_title("Mean Annual Sea Level — Lagos", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (m)")
    plt.tight_layout()
    path = PLOTS / "03_sea_level_trend.png"
    plt.savefig(path, dpi=150)
    print(f"✅ Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# PLOT 4 — Correlation Heatmap
# ══════════════════════════════════════════════════════════
def plot_correlation_heatmap():
    numeric_cols = df.select_dtypes(include=np.number).drop(
        columns=["year", "month", "day_of_year"], errors="ignore"
    )

    if numeric_cols.shape[1] < 2:
        print("⚠️  Not enough numeric columns for correlation heatmap.")
        return

    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = PLOTS / "04_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    print(f"✅ Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# PLOT 5 — Flood vs Non-Flood Rainfall Distribution
# ══════════════════════════════════════════════════════════
def plot_flood_vs_no_flood():
    if "flood_occurred" not in df.columns or df["flood_occurred"].isna().all():
        print("⚠️  No flood labels — skipping plot 5")
        return
    if "rainfall_mm" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rainfall distribution
    df[df["flood_occurred"] == 0]["rainfall_mm"].plot(
        kind="hist", bins=50, ax=axes[0], alpha=0.7,
        color="#42A5F5", label="No Flood")
    df[df["flood_occurred"] == 1]["rainfall_mm"].plot(
        kind="hist", bins=50, ax=axes[0], alpha=0.7,
        color="#EF5350", label="Flood")
    axes[0].set_title("Rainfall Distribution: Flood vs No Flood")
    axes[0].set_xlabel("Rainfall (mm)")
    axes[0].legend()

    # 7-day cumulative rain
    if "rain_7day" in df.columns:
        df[df["flood_occurred"] == 0]["rain_7day"].plot(
            kind="hist", bins=50, ax=axes[1], alpha=0.7,
            color="#42A5F5", label="No Flood")
        df[df["flood_occurred"] == 1]["rain_7day"].plot(
            kind="hist", bins=50, ax=axes[1], alpha=0.7,
            color="#EF5350", label="Flood")
        axes[1].set_title("7-Day Cumulative Rain: Flood vs No Flood")
        axes[1].set_xlabel("7-Day Rainfall (mm)")
        axes[1].legend()

    plt.tight_layout()
    path = PLOTS / "05_flood_vs_nofloat_distributions.png"
    plt.savefig(path, dpi=150)
    print(f"✅ Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════
# PLOT 6 — Flood Events Timeline
# ══════════════════════════════════════════════════════════
def plot_flood_timeline():
    if "flood_occurred" not in df.columns or df["flood_occurred"].isna().all():
        print("⚠️  No flood labels — skipping plot 6")
        return

    annual_floods = df.groupby("year")["flood_occurred"].sum().reset_index()
    annual_floods.columns = ["year", "flood_days"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(annual_floods["year"], annual_floods["flood_days"],
           color="#EF5350", alpha=0.85)
    ax.set_title("Annual Flood Days in Nigeria", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Flood Days")
    plt.tight_layout()
    path = PLOTS / "06_annual_flood_days.png"
    plt.savefig(path, dpi=150)
    print(f"✅ Saved: {path}")
    plt.close()


# ── Run all plots ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎨 Generating EDA plots...\n")
    plot_annual_rainfall()
    plot_monthly_seasonality()
    plot_sea_level_trend()
    plot_correlation_heatmap()
    plot_flood_vs_no_flood()
    plot_flood_timeline()
    print(f"\n✅ All plots saved to: {PLOTS}")
    print("   ➡️  Next step: Run 04_train_model.py")

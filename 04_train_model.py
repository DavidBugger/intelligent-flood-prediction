"""
=============================================================
INTELLIGENT FLOOD PREDICTION - NIGERIAN COASTAL CITIES
Step 4: Model Training — XGBoost + LSTM
=============================================================
Trains two models:
  Model A — XGBoost (tabular baseline, fast & interpretable)
  Model B — LSTM    (deep learning, captures time patterns)

Saves both models and prints evaluation metrics.
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("flood_project_data")
PROC     = BASE_DIR / "processed"
MODELS   = BASE_DIR / "models"
PLOTS    = BASE_DIR / "outputs/plots"

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
df = pd.read_csv(PROC / "master_dataset.csv", parse_dates=["date"])

# Drop rows without flood labels
if df["flood_occurred"].isna().all():
    print("   No flood labels found. Cannot train model yet.")
    print("   Please add Dartmouth/EM-DAT data and re-run 02_preprocess_data.py")
    exit()

df = df.dropna(subset=["flood_occurred"])
print(f"   Dataset loaded: {len(df)} rows with flood labels")
print(f"   Class balance: {df['flood_occurred'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────
# FEATURE SELECTION
# ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Rainfall features
    "rainfall_mm", "rain_3day", "rain_7day", "rain_14day", "rain_30day",
    "heavy_rain_flag", "regional_rainfall_mm",
    # Sea level features
    "sea_level_m_lagos", "sea_level_m_port_harcourt",
    "tide_level_m", "tide_7day_mean", "tide_7day_max", "high_tide_flag",
    # Combined risk
    "combined_risk_flag",
    # Calendar
    "month", "day_of_year", "wet_season",
]

# Only keep columns that exist in the dataframe
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
TARGET_COL   = "flood_occurred"

print(f"\n🔧 Using {len(FEATURE_COLS)} features:")
print(f"   {FEATURE_COLS}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].astype(int)

# Fill remaining NaNs with column median
X = X.fillna(X.median())

# ──────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (chronological — no data leakage!)
# ──────────────────────────────────────────────────────────
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════
# MODEL A — XGBoost Classifier
# ══════════════════════════════════════════════════════════
def train_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed. Run: pip install xgboost")
        return None

    print("\n" + "=" * 50)
    print("  Training Model A — XGBoost")
    print("=" * 50)

    # Class weight to handle imbalanced data (floods are rare!)
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,   # handles class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50,
    )

    # Evaluation
    y_pred      = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n  XGBoost Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))
    print(f"   ROC-AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

    # Feature importance plot
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    importance.plot(kind="barh", ax=ax, color="#1565C0", alpha=0.8)
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(PLOTS / "07_xgboost_feature_importance.png", dpi=150)
    plt.close()

    # Save model
    with open(MODELS / "xgboost_flood_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(" XGBoost model saved: models/xgboost_flood_model.pkl")

    return model, y_pred_prob


# ══════════════════════════════════════════════════════════
# MODEL B — LSTM Neural Network
# ══════════════════════════════════════════════════════════
def create_sequences(X, y, window=14):
    """Turns daily data into 14-day lookback sequences for LSTM."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y.iloc[i])
    return np.array(Xs), np.array(ys)


def train_lstm():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                              BatchNormalization)
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        print(" TensorFlow not installed. Run: pip install tensorflow")
        return None

    print("\n" + "=" * 50)
    print(" Training Model B — LSTM Neural Network")
    print("=" * 50)

    WINDOW = 14  # 14-day lookback window

    # Create sequences
    X_seq_train, y_seq_train = create_sequences(
        pd.DataFrame(X_train_scaled, columns=FEATURE_COLS),
        y_train.reset_index(drop=True), WINDOW
    )
    X_seq_test, y_seq_test = create_sequences(
        pd.DataFrame(X_test_scaled, columns=FEATURE_COLS),
        y_test.reset_index(drop=True), WINDOW
    )

    print(f"   Input shape: {X_seq_train.shape}  "
          f"(samples, timesteps, features)")

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(WINDOW, len(FEATURE_COLS))),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    # Class weights for imbalance - more aggressive weighting
    neg = (y_seq_train == 0).sum()
    pos = (y_seq_train == 1).sum()
    class_weight = {0: 1.0, 1: (neg / pos) * 2.0}  # Double the weight for floods
    print(f"   Class weights: {class_weight}")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=15,
                      restore_best_weights=True, mode="max"),
        ModelCheckpoint(str(MODELS / "lstm_best.h5"),
                        monitor="val_auc", save_best_only=True, mode="max"),
    ]

    history = model.fit(
        X_seq_train, y_seq_train,
        validation_data=(X_seq_test, y_seq_test),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation
    y_pred_prob = model.predict(X_seq_test).flatten()
    
    # Handle NaN values in predictions
    nan_mask = ~np.isnan(y_pred_prob)
    if nan_mask.sum() < len(y_pred_prob):
        print(f" Warning: {len(y_pred_prob) - nan_mask.sum()} NaN predictions found")
        y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.0)  # Replace NaN with 0.0
    
    y_pred = (y_pred_prob > 0.3).astype(int)  # Lower threshold for better recall

    print("\n LSTM Classification Report:")
    print(classification_report(y_seq_test, y_pred,
                                 target_names=["No Flood", "Flood"]))
    
    # Only calculate AUC if we have valid predictions
    if len(np.unique(y_seq_test)) > 1 and len(np.unique(y_pred_prob)) > 1:
        try:
            auc_score = roc_auc_score(y_seq_test, y_pred_prob)
            print(f"   ROC-AUC Score: {auc_score:.4f}")
        except Exception as e:
            print(f"   ROC-AUC Score: Could not calculate ({e})")
    else:
        print(f"   ROC-AUC Score: Not available (insufficient class variation)")

    # Training history plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("LSTM Training Loss")
    axes[0].legend()

    axes[1].plot(history.history["auc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("LSTM AUC Score")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "08_lstm_training_history.png", dpi=150)
    plt.close()

    print(" LSTM model saved: models/lstm_best.h5")
    return model


# ──────────────────────────────────────────────────────────
# SAVE SCALER FOR INFERENCE
# ──────────────────────────────────────────────────────────
with open(MODELS / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

meta = {"feature_cols": FEATURE_COLS, "window": 14}
with open(MODELS / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    xgb_model, _ = train_xgboost()
    lstm_model   = train_lstm()
    print("\n🎉 All models trained!")
    print("   Next step: Run 05_early_warning_system.py")

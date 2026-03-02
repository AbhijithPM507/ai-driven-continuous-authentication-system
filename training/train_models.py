import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ===============================
# CONFIG
# ===============================

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CONTAMINATION = 0.05   # assume 5% anomaly (safe demo setting)

# ===============================
# LOAD DATASETS
# ===============================

key_df = pd.read_csv("data/processed/keystroke_windows.csv")
mouse_df = pd.read_csv("data/processed/mouse_windows.csv")
click_df = pd.read_csv("data/processed/click_windows.csv")

print("Loaded datasets:")
print("Keystroke windows:", len(key_df))
print("Mouse windows:", len(mouse_df))
print("Click windows:", len(click_df))

# ===============================
# FEATURE ORDER (STRICT)
# ===============================

KEY_COLS = [
    "mean_hold",
    "std_hold",
    "mean_flight",
    "std_flight",
    "typing_speed"
]

MOUSE_COLS = [
    "mean_velocity",
    "std_velocity",
    "mean_acceleration",
    "std_acceleration",
    "movement_density"
]

CLICK_COLS = [
    "mean_interval",
    "std_interval",
    "click_rate"
]

# ===============================
# TRAIN FUNCTION
# ===============================

def train_modality(df, feature_cols, model_name):

    X = df[feature_cols].copy()

    # Replace NaN safely
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(X_scaled)

    # Save model + scaler
    joblib.dump(model, f"{MODEL_DIR}/{model_name}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{model_name}_scaler.pkl")

    print(f"✅ {model_name} model trained & saved")

    # Return anomaly scores for fusion training
    scores = -model.decision_function(X_scaled)
    return scores.reshape(-1, 1)


# ===============================
# TRAIN EACH MODALITY
# ===============================

key_scores = train_modality(key_df, KEY_COLS, "keystroke")
mouse_scores = train_modality(mouse_df, MOUSE_COLS, "mouse")
click_scores = train_modality(click_df, CLICK_COLS, "click")

# ===============================
# ALIGN SCORES (SAFE TRUNCATION)
# ===============================

min_len = min(len(key_scores), len(mouse_scores), len(click_scores))

key_scores = key_scores[:min_len]
mouse_scores = mouse_scores[:min_len]
click_scores = click_scores[:min_len]

fusion_scores = np.hstack([
    key_scores,
    mouse_scores,
    click_scores
])

# ===============================
# FUSION SCALER
# ===============================

fusion_scaler = MinMaxScaler()
fusion_scaled = fusion_scaler.fit_transform(fusion_scores)

joblib.dump(fusion_scaler, f"{MODEL_DIR}/fusion_scaler.pkl")

print("✅ Fusion scaler saved")

# ===============================
# WEIGHTED FUSION
# ===============================

final_scores = (
    0.5 * fusion_scaled[:, 0] +
    0.3 * fusion_scaled[:, 1] +
    0.2 * fusion_scaled[:, 2]
)

# ===============================
# THRESHOLD (95th Percentile)
# ===============================

threshold = np.percentile(final_scores, 95)

joblib.dump(threshold, f"{MODEL_DIR}/fusion_threshold.pkl")

print("✅ Fusion threshold saved")
print("Threshold value:", threshold)

print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY")
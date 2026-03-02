import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
STATUS_FILE = os.path.join(BASE_DIR, "runtime_status.json")

WINDOW_SIZE = 60

def get_latest_window(filepath, time_col):
    try:
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_csv(filepath)
        if df.empty:
            return df
        latest_time = df[time_col].max()
        df_window = df[df[time_col] >= (latest_time - WINDOW_SIZE)]
        return df_window
    except Exception:
        return pd.DataFrame()

def extract_key_features(df):
    if df.empty or len(df) < 10:
        return None
    return pd.DataFrame([{
        "mean_hold": df["hold_time"].mean(),
        "std_hold": df["hold_time"].std(),
        "mean_flight": df["flight_time"].mean(),
        "std_flight": df["flight_time"].std(),
        "typing_speed": len(df) / WINDOW_SIZE
    }]).fillna(0)

def extract_mouse_features(df):
    if df.empty or len(df) < 20:
        return None
    return pd.DataFrame([{
        "mean_velocity": df["velocity"].mean(),
        "std_velocity": df["velocity"].std(),
        "mean_acceleration": df["acceleration"].mean(),
        "std_acceleration": df["acceleration"].std(),
        "movement_density": len(df) / WINDOW_SIZE
    }]).fillna(0)

def extract_click_features(df):
    if df.empty or len(df) < 5:
        return None
    return pd.DataFrame([{
        "mean_interval": df["interval"].mean(),
        "std_interval": df["interval"].std(),
        "click_rate": len(df) / WINDOW_SIZE
    }]).fillna(0)

def run_monitor():
    print("[INFO] Starting real-time monitor...")
    
    while not os.path.exists(os.path.join(MODEL_DIR, "fusion_threshold.pkl")):
        print("[INFO] Waiting for models to be trained...")
        time.sleep(2)
        
    key_model = joblib.load(os.path.join(MODEL_DIR, "keystroke_model.pkl"))
    mouse_model = joblib.load(os.path.join(MODEL_DIR, "mouse_model.pkl"))
    click_model = joblib.load(os.path.join(MODEL_DIR, "click_model.pkl"))
    
    key_scaler = joblib.load(os.path.join(MODEL_DIR, "keystroke_scaler.pkl"))
    mouse_scaler = joblib.load(os.path.join(MODEL_DIR, "mouse_scaler.pkl"))
    click_scaler = joblib.load(os.path.join(MODEL_DIR, "click_scaler.pkl"))
    
    fusion_scaler = joblib.load(os.path.join(MODEL_DIR, "fusion_scaler.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "fusion_threshold.pkl"))

    streak = 0
    
    last_key_score = 0.0
    last_mouse_score = 0.0
    last_click_score = 0.0
    
    while True:
        try:
            key_raw = get_latest_window(os.path.join(RAW_DIR, "keystroke.csv"), "press_time")
            mouse_raw = get_latest_window(os.path.join(RAW_DIR, "mouse_move.csv"), "timestamp")
            click_raw = get_latest_window(os.path.join(RAW_DIR, "mouse_click.csv"), "timestamp")
            
            key_f = extract_key_features(key_raw)
            mouse_f = extract_mouse_features(mouse_raw)
            click_f = extract_click_features(click_raw)
            
            if key_f is not None:
                scaled = key_scaler.transform(key_f)
                last_key_score = float(-key_model.decision_function(scaled)[0])
                
            if mouse_f is not None:
                scaled = mouse_scaler.transform(mouse_f)
                last_mouse_score = float(-mouse_model.decision_function(scaled)[0])
                
            if click_f is not None:
                scaled = click_scaler.transform(click_f)
                last_click_score = float(-click_model.decision_function(scaled)[0])
                
            combined = np.array([[last_key_score, last_mouse_score, last_click_score]])
            normalized = fusion_scaler.transform(combined)[0]
            
            fusion_score = float(0.5 * normalized[0] + 0.3 * normalized[1] + 0.2 * normalized[2])
            
            if fusion_score > threshold:
                streak += 1
            else:
                streak = 0
                
            status_text = "INTRUDER" if streak >= 3 else "AUTHENTICATED"
            
            status_data = {
                "status": status_text,
                "fusion_score": round(fusion_score, 4),
                "threshold": round(float(threshold), 4),
                "streak": streak,
                "key_score": round(last_key_score, 4),
                "mouse_score": round(last_mouse_score, 4),
                "shortcut_score": 0.0
            }
            
            temp_file = STATUS_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(status_data, f, indent=2)
            os.replace(temp_file, STATUS_FILE)
            
        except Exception as e:
            pass
            
        time.sleep(0.2)

if __name__ == "__main__":
    run_monitor()

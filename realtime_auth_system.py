import pandas as pd
import numpy as np
import joblib
import time
import os
import signal
import sys
import atexit
from datetime import datetime
from pynput import keyboard, mouse

# ==============================
# CONFIG
# ==============================

WINDOW_SIZE = 30      # seconds for feature window
CHECK_INTERVAL = 10   # seconds between predictions
LOG_FILE = "live_session.csv"

MODEL_FILE = "behavior_model.pkl"
SCALER_FILE = "scaler.pkl"

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ==============================
# INIT CSV
# ==============================

with open(LOG_FILE, mode="w") as f:
    f.write("timestamp,key_dwell,mouse_x,mouse_y,scroll_dx,scroll_dy,idle_seconds,event_type\n")

# ==============================
# GLOBAL VARIABLES
# ==============================

key_press_times = {}
last_activity_time = time.time()
current_mouse_x = 0
current_mouse_y = 0
running = True

# ==============================
# CLEANUP
# ==============================

def cleanup():
    global running
    running = False
    time.sleep(0.5)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print("\nSession file deleted.")

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

# ==============================
# WRITE EVENT
# ==============================

def write_event(key_dwell=0, scroll_dx=0, scroll_dy=0, event_type=""):

    global last_activity_time

    if not running:
        return

    timestamp = datetime.now()
    idle_seconds = time.time() - last_activity_time
    last_activity_time = time.time()

    with open(LOG_FILE, mode="a") as f:
        f.write(f"{timestamp},{key_dwell},{current_mouse_x},{current_mouse_y},{scroll_dx},{scroll_dy},{idle_seconds},{event_type}\n")

# ==============================
# KEYBOARD
# ==============================

def on_press(key):
    key_press_times[key] = time.time()

def on_release(key):
    if key in key_press_times:
        dwell = time.time() - key_press_times[key]
        write_event(key_dwell=dwell, event_type="key")
        del key_press_times[key]

# ==============================
# MOUSE
# ==============================

def on_move(x, y):
    global current_mouse_x, current_mouse_y
    current_mouse_x = x
    current_mouse_y = y
    write_event(event_type="move")

def on_click(x, y, button, pressed):
    if pressed:
        write_event(event_type="click")

def on_scroll(x, y, dx, dy):
    write_event(scroll_dx=dx, scroll_dy=dy, event_type="scroll")

# ==============================
# REAL-TIME FEATURE EXTRACTION
# ==============================

def extract_features(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    # Keep only last 30 sec
    df = df[df["time_seconds"] >= df["time_seconds"].max() - WINDOW_SIZE]

    if len(df) < 10:
        return None

    df["dx"] = df["mouse_x"].diff()
    df["dy"] = df["mouse_y"].diff()
    df["dt"] = df["time_seconds"].diff()

    df["distance"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["dt"] = df["dt"].replace(0, np.nan)

    df["mouse_speed"] = df["distance"] / df["dt"]
    df["mouse_acc"] = df["mouse_speed"].diff() / df["dt"]
    df["scroll_int"] = np.sqrt(df["scroll_dx"]**2 + df["scroll_dy"]**2)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    features = {
        "key_dwell_mean": df["key_dwell"].mean(),
        "key_dwell_std": df["key_dwell"].std(),
        "mouse_speed_mean": df["mouse_speed"].mean(),
        "mouse_speed_std": df["mouse_speed"].std(),
        "mouse_acc_mean": df["mouse_acc"].mean(),
        "mouse_acc_std": df["mouse_acc"].std(),
        "distance_sum": df["distance"].sum(),
        "scroll_int_mean": df["scroll_int"].mean(),
        "scroll_int_std": df["scroll_int"].std(),
        "idle_mean": df["idle_seconds"].mean(),
        "idle_max": df["idle_seconds"].max(),
        "event_count": len(df)
    }

    return pd.DataFrame([features])

# ==============================
# AUTHENTICATION LOOP
# ==============================

def authentication_loop():

    intruder_count = 0

    while running:

        try:
            df = pd.read_csv(LOG_FILE)

            features = extract_features(df)

            if features is not None:
                scaled = scaler.transform(features)
                prediction = model.predict(scaled)[0]

                if prediction == 1:
                    intruder_count += 1
                    print("⚠ Intruder detected")
                else:
                    intruder_count = 0
                    print("✅ Genuine user")

                if intruder_count >= 3:
                    print("🔒 SYSTEM LOCKED")
                    sys.exit(0)

        except:
            pass

        time.sleep(CHECK_INTERVAL)

# ==============================
# START LISTENERS
# ==============================

keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)

keyboard_listener.start()
mouse_listener.start()

print("Real-time authentication system started.")
print("Press Ctrl+C or close terminal to stop.")

authentication_loop()

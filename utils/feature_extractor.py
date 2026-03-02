import pandas as pd
import numpy as np
from datetime import timedelta

WINDOW_SIZE = 60

def extract_features(df):

    if df.empty:
        return None, None, None

    # Convert timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    df = df.sort_values("timestamp")

    end_time = df["timestamp"].max()
    start_time = end_time - timedelta(seconds=WINDOW_SIZE)

    window = df[df["timestamp"] >= start_time].copy()

    if len(window) < 10:
        return None, None, None

    # =============================
    # KEYSTROKE FEATURES
    # =============================
    key_df = window[window["event_type"] == "key"].copy()

    if len(key_df) >= 10:

        key_df["flight"] = key_df["timestamp"].diff().dt.total_seconds()

        key_features = pd.DataFrame([{
            "mean_hold": key_df["key_dwell"].mean(),
            "std_hold": key_df["key_dwell"].std(),
            "mean_flight": key_df["flight"].mean(),
            "std_flight": key_df["flight"].std(),
            "typing_speed": len(key_df) / WINDOW_SIZE
        }])

    else:
        key_features = None

    # =============================
    # MOUSE MOVE FEATURES
    # =============================
    mouse_df = window[window["event_type"] == "move"].copy()

    if len(mouse_df) >= 20:

        mouse_df["dx"] = mouse_df["mouse_x"].diff()
        mouse_df["dy"] = mouse_df["mouse_y"].diff()
        mouse_df["dt"] = mouse_df["timestamp"].diff().dt.total_seconds()

        mouse_df["dt"] = mouse_df["dt"].replace(0, 0.0001)

        mouse_df["velocity"] = np.sqrt(
            mouse_df["dx"]**2 + mouse_df["dy"]**2
        ) / mouse_df["dt"]

        mouse_df["acceleration"] = mouse_df["velocity"].diff() / mouse_df["dt"]

        mouse_features = pd.DataFrame([{
            "mean_velocity": mouse_df["velocity"].mean(),
            "std_velocity": mouse_df["velocity"].std(),
            "mean_acceleration": mouse_df["acceleration"].mean(),
            "std_acceleration": mouse_df["acceleration"].std(),
            "movement_density": len(mouse_df) / WINDOW_SIZE
        }])

    else:
        mouse_features = None

    # =============================
    # CLICK FEATURES
    # =============================
    click_df = window[window["event_type"] == "click"].copy()

    if len(click_df) >= 5:

        click_df["interval"] = click_df["timestamp"].diff().dt.total_seconds()

        click_features = pd.DataFrame([{
            "mean_interval": click_df["interval"].mean(),
            "std_interval": click_df["interval"].std(),
            "click_rate": len(click_df) / WINDOW_SIZE
        }])

    else:
        click_features = None

    # Replace NaN safely
    if key_features is not None:
        key_features = key_features.fillna(0)

    if mouse_features is not None:
        mouse_features = mouse_features.fillna(0)

    if click_features is not None:
        click_features = click_features.fillna(0)

    return key_features, mouse_features, click_features
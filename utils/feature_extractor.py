# utils/feature_extractor.py

import pandas as pd
import numpy as np
from datetime import timedelta

WINDOW_SIZE = 60  # seconds

MIN_KEY = 10
MIN_MOUSE = 20
MIN_CLICK = 5


def extract_features():

    try:
        df = pd.read_csv("data/live_session.csv")
    except:
        return None

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    now = df["timestamp"].max()
    window_start = now - timedelta(seconds=WINDOW_SIZE)

    df = df[df["timestamp"] >= window_start].copy()

    if len(df) == 0:
        return None

    # =========================
    # KEYSTROKE FEATURES
    # =========================
    key_df = df[df["event_type"] == "key"].copy()

    if len(key_df) < MIN_KEY:
        return None

    key_df["flight"] = key_df["timestamp"].diff().dt.total_seconds()

    mean_hold = key_df["key_dwell"].mean()
    std_hold = key_df["key_dwell"].std()
    mean_flight = key_df["flight"].mean()
    std_flight = key_df["flight"].std()
    typing_speed = len(key_df) / WINDOW_SIZE

    keystroke_features = [[
        mean_hold,
        std_hold if not np.isnan(std_hold) else 0,
        mean_flight if not np.isnan(mean_flight) else 0,
        std_flight if not np.isnan(std_flight) else 0,
        typing_speed
    ]]

    # =========================
    # MOUSE FEATURES
    # =========================
    mouse_df = df[df["event_type"] == "move"].copy()

    if len(mouse_df) < MIN_MOUSE:
        return None

    mouse_df["dx"] = mouse_df["mouse_x"].diff()
    mouse_df["dy"] = mouse_df["mouse_y"].diff()
    mouse_df["dt"] = mouse_df["timestamp"].diff().dt.total_seconds()

    mouse_df["velocity"] = np.sqrt(mouse_df["dx"]**2 + mouse_df["dy"]**2) / mouse_df["dt"]
    mouse_df["acceleration"] = mouse_df["velocity"].diff() / mouse_df["dt"]

    mean_velocity = mouse_df["velocity"].mean()
    std_velocity = mouse_df["velocity"].std()
    mean_acceleration = mouse_df["acceleration"].mean()
    std_acceleration = mouse_df["acceleration"].std()

    movement_density = len(mouse_df) / WINDOW_SIZE

    mouse_features = [[
        mean_velocity if not np.isnan(mean_velocity) else 0,
        std_velocity if not np.isnan(std_velocity) else 0,
        mean_acceleration if not np.isnan(mean_acceleration) else 0,
        std_acceleration if not np.isnan(std_acceleration) else 0,
        movement_density
    ]]

    # =========================
    # CLICK FEATURES
    # =========================
    click_df = df[df["event_type"] == "click"].copy()

    if len(click_df) < MIN_CLICK:
        return None

    click_df["interval"] = click_df["timestamp"].diff().dt.total_seconds()

    mean_interval = click_df["interval"].mean()
    std_interval = click_df["interval"].std()
    click_rate = len(click_df) / WINDOW_SIZE

    click_features = [[
        mean_interval if not np.isnan(mean_interval) else 0,
        std_interval if not np.isnan(std_interval) else 0,
        click_rate
    ]]

    return {
        "keystroke": keystroke_features,
        "mouse": mouse_features,
        "click": click_features
    }

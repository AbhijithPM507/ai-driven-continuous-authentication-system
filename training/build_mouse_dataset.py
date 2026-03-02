import pandas as pd
import numpy as np
from datetime import timedelta

WINDOW_SIZE = 60

df = pd.read_csv("data/raw/mouse_move.csv")

# FIX: convert epoch seconds properly
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

df = df.sort_values("timestamp")

start = df["timestamp"].min()
end = df["timestamp"].max()

print("Mouse duration (seconds):", (end - start).total_seconds())

rows = []

while start + timedelta(seconds=WINDOW_SIZE) <= end:

    window = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] < start + timedelta(seconds=WINDOW_SIZE))
    ]

    if len(window) >= 5:   # lowered threshold for demo stability

        window = window.copy()

        window["dx"] = window["x"].diff()
        window["dy"] = window["y"].diff()
        window["dt"] = window["timestamp"].diff().dt.total_seconds()

        window["dt"] = window["dt"].replace(0, 0.0001)

        window["velocity"] = np.sqrt(window["dx"]**2 + window["dy"]**2) / window["dt"]
        window["acceleration"] = window["velocity"].diff() / window["dt"]

        rows.append({
            "mean_velocity": window["velocity"].mean(),
            "std_velocity": window["velocity"].std(),
            "mean_acceleration": window["acceleration"].mean(),
            "std_acceleration": window["acceleration"].std(),
            "movement_density": len(window) / WINDOW_SIZE
        })

    start += timedelta(seconds=WINDOW_SIZE)

pd.DataFrame(rows).to_csv("data/processed/mouse_windows.csv", index=False)

print("✅ Mouse dataset built successfully.")
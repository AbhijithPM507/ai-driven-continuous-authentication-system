import pandas as pd
from datetime import timedelta

WINDOW_SIZE = 60

df = pd.read_csv("data/raw/mouse_click.csv")

# FIX: convert epoch seconds properly
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

df = df.sort_values("timestamp")

start = df["timestamp"].min()
end = df["timestamp"].max()

print("Click duration (seconds):", (end - start).total_seconds())

rows = []

while start + timedelta(seconds=WINDOW_SIZE) <= end:

    window = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] < start + timedelta(seconds=WINDOW_SIZE))
    ]

    if len(window) >= 2:   # lowered for demo stability

        window = window.copy()
        window["interval"] = window["timestamp"].diff().dt.total_seconds()

        rows.append({
            "mean_interval": window["interval"].mean(),
            "std_interval": window["interval"].std(),
            "click_rate": len(window) / WINDOW_SIZE
        })

    start += timedelta(seconds=WINDOW_SIZE)

pd.DataFrame(rows).to_csv("data/processed/click_windows.csv", index=False)

print("✅ Click dataset built successfully.")
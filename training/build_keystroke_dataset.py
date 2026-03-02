import pandas as pd
from datetime import timedelta

WINDOW_SIZE = 60  # must match live system

# ---------------- LOAD RAW DATA ----------------
df = pd.read_csv("data/raw/keystroke.csv")

# ---------------- SAFETY CHECK ----------------
required_columns = ["press_time", "release_time"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ---------------- CONVERT TIMESTAMPS ----------------
# Convert Unix epoch seconds → datetime
df["press_time"] = pd.to_datetime(df["press_time"], unit="s")
df["release_time"] = pd.to_datetime(df["release_time"], unit="s")

# Sort chronologically
df = df.sort_values("press_time").reset_index(drop=True)

# ---------------- COMPUTE FEATURES ----------------
# Hold time (seconds)
df["hold"] = (df["release_time"] - df["press_time"]).dt.total_seconds()

# Flight time (time between consecutive key presses)
df["flight"] = df["press_time"].diff().dt.total_seconds()

# Remove first row flight NaN
df = df.dropna(subset=["flight"])

# ---------------- WINDOW SEGMENTATION ----------------
start_time = df["press_time"].min()
end_time = df["press_time"].max()

duration = (end_time - start_time).total_seconds()
print(f"Total keystroke duration: {round(duration,2)} seconds")

rows = []

while start_time + timedelta(seconds=WINDOW_SIZE) <= end_time:

    window_end = start_time + timedelta(seconds=WINDOW_SIZE)

    window = df[
        (df["press_time"] >= start_time) &
        (df["press_time"] < window_end)
    ].copy()

    # Minimum 10 keystrokes per window
    if len(window) >= 10:

        rows.append({
            "mean_hold": window["hold"].mean(),
            "std_hold": window["hold"].std(),
            "mean_flight": window["flight"].mean(),
            "std_flight": window["flight"].std(),
            "typing_speed": len(window) / WINDOW_SIZE
        })

    start_time += timedelta(seconds=WINDOW_SIZE)

# ---------------- SAVE PROCESSED DATA ----------------
output_df = pd.DataFrame(rows)

if len(output_df) == 0:
    print("⚠ WARNING: No keystroke windows created.")
else:
    output_df.to_csv("data/processed/keystroke_windows.csv", index=False)
    print(f"✅ Keystroke dataset built successfully with {len(output_df)} windows.")
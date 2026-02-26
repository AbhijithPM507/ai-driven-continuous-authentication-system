# logger.py

import os
import time
import csv
from datetime import datetime
from pynput import keyboard, mouse

# ===============================
# CONFIG
# ===============================

DATA_FOLDER = "data"
LOG_FILE = os.path.join(DATA_FOLDER, "live_session.csv")

# ===============================
# SETUP FILE
# ===============================

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "key_dwell",
            "mouse_x",
            "mouse_y",
            "scroll_dx",
            "scroll_dy",
            "idle_seconds",
            "event_type"
        ])

print("📝 Logger started...")
print("Writing to:", LOG_FILE)

# ===============================
# GLOBAL STATE
# ===============================

key_press_times = {}
last_activity_time = time.time()
current_mouse_x = 0
current_mouse_y = 0


# ===============================
# WRITE EVENT FUNCTION
# ===============================

def write_event(key_dwell=0, scroll_dx=0, scroll_dy=0, event_type=""):

    global last_activity_time

    timestamp = datetime.now()
    idle_seconds = time.time() - last_activity_time
    last_activity_time = time.time()

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            key_dwell,
            current_mouse_x,
            current_mouse_y,
            scroll_dx,
            scroll_dy,
            idle_seconds,
            event_type
        ])


# ===============================
# KEYBOARD EVENTS
# ===============================

def on_press(key):
    key_press_times[key] = time.time()


def on_release(key):
    if key in key_press_times:
        dwell = time.time() - key_press_times[key]
        write_event(key_dwell=dwell, event_type="key")
        del key_press_times[key]


# ===============================
# MOUSE EVENTS
# ===============================

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


# ===============================
# START LISTENERS
# ===============================

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll
)

keyboard_listener.start()
mouse_listener.start()

print("⌨️ Keyboard and 🖱 Mouse monitoring started.")
print("Press CTRL+C to stop.\n")

# Keep alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Logger stopped.")

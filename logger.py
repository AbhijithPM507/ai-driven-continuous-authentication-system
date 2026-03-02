import csv
import os
import time
from datetime import datetime
from pynput import keyboard, mouse

DATA_FILE = "data/live_session.csv"

os.makedirs("data", exist_ok=True)

# Create file if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "event_type",
            "key_dwell",
            "mouse_x",
            "mouse_y"
        ])

key_press_times = {}
current_mouse_x = 0
current_mouse_y = 0


# ==========================
# WRITE FUNCTION
# ==========================
def write_event(event_type, key_dwell=0):

    timestamp = time.time()

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            event_type,
            key_dwell,
            current_mouse_x,
            current_mouse_y
        ])


# ==========================
# KEYBOARD EVENTS
# ==========================
def on_press(key):
    key_press_times[key] = time.time()


def on_release(key):
    if key in key_press_times:
        dwell = time.time() - key_press_times[key]
        write_event("key", dwell)
        del key_press_times[key]


# ==========================
# MOUSE EVENTS
# ==========================
def on_move(x, y):
    global current_mouse_x, current_mouse_y
    current_mouse_x = x
    current_mouse_y = y
    write_event("move")


def on_click(x, y, button, pressed):
    if pressed:
        write_event("click")


# ==========================
# START LISTENERS
# ==========================
print("📡 Logger started...")

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click
)

keyboard_listener.start()
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()
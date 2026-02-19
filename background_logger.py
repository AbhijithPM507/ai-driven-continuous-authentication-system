"""
Privacy-Preserving Behaviour Logger (Continuous Mode)
----------------------------------------------------
Logs ONLY behavioural biometrics:
- Keyboard timing (no actual keys stored)
- Mouse movement, click, scroll
- Idle time

Does NOT log:
- Typed text
- Application names
- Window titles
- Websites
"""

import os
import csv
import time
import uuid
import threading
from datetime import datetime
from pynput import keyboard, mouse

# ---------------- CONFIG ----------------
USER_ID = "user_01"
LOG_DIR = "logs"
IDLE_THRESHOLD = 3.0  # seconds of inactivity considered idle

# ---------------- GLOBAL STATE ----------------
session_id = str(uuid.uuid4())[:8]
stop_flag = False
last_activity_time = time.time()
key_press_times = {}

# ---------------- UTILITIES ----------------
def now_iso():
    return datetime.now().isoformat(timespec="milliseconds")

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def log_file_path():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{USER_ID}_{date_str}_{session_id}_private.csv")

def safe_key_category(k):
    """Return only key category, not actual key."""
    try:
        if hasattr(k, "char") and k.char is not None:
            if k.char.isalnum():
                return "ALNUM"
            return "CHAR_OTHER"
        return str(k).replace("Key.", "").upper()
    except:
        return "UNKNOWN"

# ---------------- CSV LOGGER ----------------
class CSVLogger:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        if os.stat(path).st_size == 0:
            self.writer.writerow([
                "timestamp", "user_id", "session_id", "event_type",
                "key_category", "key_dwell_ms",
                "mouse_x", "mouse_y", "mouse_event",
                "scroll_dx", "scroll_dy",
                "idle_seconds"
            ])
            self.file.flush()

    def write(self, row):
        with self.lock:
            self.writer.writerow(row)
            self.file.flush()

    def close(self):
        with self.lock:
            self.file.close()

# ---------------- IDLE TRACKER ----------------
def idle_worker(logger):
    global last_activity_time, stop_flag
    idle_logged = False

    while not stop_flag:
        idle_time = time.time() - last_activity_time

        if idle_time >= IDLE_THRESHOLD and not idle_logged:
            idle_logged = True
            logger.write([now_iso(), USER_ID, session_id, "IDLE_START",
                          "", "", "", "", "", "", "", round(idle_time, 3)])

        elif idle_time < IDLE_THRESHOLD and idle_logged:
            idle_logged = False
            logger.write([now_iso(), USER_ID, session_id, "IDLE_END",
                          "", "", "", "", "", "", "", round(idle_time, 3)])

        time.sleep(0.25)

# ---------------- KEYBOARD ----------------
def on_key_press(key, logger):
    global last_activity_time
    last_activity_time = time.time()

    kid = safe_key_category(key)
    key_press_times[kid] = time.time()

    logger.write([now_iso(), USER_ID, session_id, "KEY_PRESS",
                  kid, "", "", "", "", "", "", ""])

def on_key_release(key, logger):
    global last_activity_time
    last_activity_time = time.time()

    kid = safe_key_category(key)
    t0 = key_press_times.get(kid)
    dwell_ms = int((time.time() - t0) * 1000) if t0 else ""

    logger.write([now_iso(), USER_ID, session_id, "KEY_RELEASE",
                  kid, dwell_ms, "", "", "", "", "", ""])

# ---------------- MOUSE ----------------
def on_move(x, y, logger):
    global last_activity_time
    last_activity_time = time.time()
    logger.write([now_iso(), USER_ID, session_id, "MOUSE_MOVE",
                  "", "", x, y, "MOVE", "", "", ""])

def on_click(x, y, button, pressed, logger):
    global last_activity_time
    last_activity_time = time.time()
    evt = f"{str(button).replace('Button.', '').upper()}_{'DOWN' if pressed else 'UP'}"
    logger.write([now_iso(), USER_ID, session_id, "MOUSE_CLICK",
                  "", "", x, y, evt, "", "", ""])

def on_scroll(x, y, dx, dy, logger):
    global last_activity_time
    last_activity_time = time.time()
    logger.write([now_iso(), USER_ID, session_id, "MOUSE_SCROLL",
                  "", "", x, y, "SCROLL", dx, dy, ""])

# ---------------- TERMINAL EXIT LISTENER ----------------
def exit_listener():
    global stop_flag
    while True:
        cmd = input()
        if cmd.strip().lower() == "exit":
            stop_flag = True
            break

# ---------------- MAIN ----------------
def main():
    ensure_log_dir()
    path = log_file_path()
    logger = CSVLogger(path)

    print("==============================================")
    print(" Behaviour Logger Running (Privacy Mode) 🔒")
    print(f" User ID    : {USER_ID}")
    print(f" Session ID : {session_id}")
    print(f" Log File   : {path}")
    print("----------------------------------------------")
    print("Type 'exit' and press ENTER here to stop.")
    print("==============================================")

    # Start idle thread
    threading.Thread(target=idle_worker, args=(logger,), daemon=True).start()

    # Start exit listener thread
    threading.Thread(target=exit_listener, daemon=True).start()

    # Start keyboard & mouse listeners
    kb_listener = keyboard.Listener(
        on_press=lambda k: on_key_press(k, logger),
        on_release=lambda k: on_key_release(k, logger)
    )
    ms_listener = mouse.Listener(
        on_move=lambda x, y: on_move(x, y, logger),
        on_click=lambda x, y, button, pressed: on_click(x, y, button, pressed, logger),
        on_scroll=lambda x, y, dx, dy: on_scroll(x, y, dx, dy, logger)
    )

    kb_listener.start()
    ms_listener.start()

    # Keep running until exit typed
    while not stop_flag:
        time.sleep(0.5)

    kb_listener.stop()
    ms_listener.stop()
    logger.write([now_iso(), USER_ID, session_id, "SESSION_END", "", "", "", "", "", "", "", ""])
    logger.close()

    print("\n✅ Logger stopped safely.")
    print(f"Saved log file: {path}")

if __name__ == "__main__":
    main()

import os
from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / 'data'
MODELS_DIR = APP_DIR / 'models' / 'saved'
WEB_DIR = APP_DIR / 'web'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class DesktopConfig:
    DATABASE_PATH = str(DATA_DIR / 'locksy.db')
    DATABASE_KEY = os.environ.get('LOCKSY_DB_KEY') or 'default-locksy-key-change-me'

    MODELS_BASE_PATH = str(MODELS_DIR)
    MODEL_RETRAIN_THRESHOLD = 0.3

    WINDOW_SIZE = 30
    ANOMALY_SCORE_THRESHOLD = 0.8
    CONFIDENCE_THRESHOLD = 0.7
    CONSECUTIVE_ANOMALIES_LIMIT = 3

    GRU_SEQUENCE_LENGTH = 50
    GRU_HIDDEN_UNITS = 64
    AUTOENCODER_ENCODING_DIM = 32
    ANOMALY_THRESHOLD = 0.15
    DRIFT_DETECTION_WINDOW = 100
    DRIFT_ALPHA = 0.05
    DRIFT_MIN_SAMPLES = 30
    BEHAVIORAL_CHANGE_THRESHOLD = 0.25

    KEYSTROKE_BUFFER_SIZE = 1000
    MOUSE_BUFFER_SIZE = 2000
    FEATURE_UPDATE_INTERVAL = 5

    EEL_PORT = 8080

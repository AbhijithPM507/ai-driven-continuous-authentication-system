import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Database Configuration
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or 'database/auth_system.db'
    
    # Model Configuration
    MODELS_BASE_PATH = os.environ.get('MODELS_PATH') or 'models/saved'
    MODEL_RETRAIN_THRESHOLD = 0.3  # Retrain if accuracy drops below 70%
    
    # Behavioral Analysis Configuration
    WINDOW_SIZE = 10  # seconds
    MIN_CALIBRATION_TIME = 30  # 30 seconds minimum calibration
    KEYSTROKE_FEATURES = [
        'key_hold_time', 'flight_time', 'typing_speed', 
        'pause_variance', 'digraph_timing', 'trigraph_timing'
    ]
    MOUSE_FEATURES = [
        'velocity', 'acceleration', 'jerk', 'curvature',
        'click_duration', 'dwell_time', 'direction_changes'
    ]
    
    # ML Model Configuration
    GRU_SEQUENCE_LENGTH = 10
    GRU_HIDDEN_UNITS = 64
    AUTOENCODER_ENCODING_DIM = 32
    ANOMALY_THRESHOLD = 0.15
    DRIFT_DETECTION_WINDOW = 20
    
    # Authentication Thresholds
    CONFIDENCE_THRESHOLD = 0.6
    ANOMALY_SCORE_THRESHOLD = 0.36
    CONSECUTIVE_ANOMALIES_LIMIT = 3  # adjusted for fast typing tolerance
    
    # WebSocket Configuration
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    SOCKETIO_LOGGER = False
    SOCKETIO_ENGINEIO_LOGGER = False
    
    # Security Configuration
    BCRYPT_LOG_ROUNDS = 12
    SESSION_TIMEOUT = timedelta(hours=8)
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=15)
    
    # Feature Extraction Configuration
    KEYSTROKE_BUFFER_SIZE = 1000
    MOUSE_BUFFER_SIZE = 2000
    FEATURE_UPDATE_INTERVAL = 5  # seconds
    
    # Drift Detection Configuration
    DRIFT_ALPHA = 0.05  # Statistical significance level
    DRIFT_MIN_SAMPLES = 30
    BEHAVIORAL_CHANGE_THRESHOLD = 0.25
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    BCRYPT_LOG_ROUNDS = 14

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_PATH = ':memory:'  # In-memory database for testing
    BCRYPT_LOG_ROUNDS = 4  # Faster for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
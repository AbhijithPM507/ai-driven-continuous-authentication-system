# Behavioral Authentication System

An advanced continuous authentication system using behavioral biometrics and machine learning to provide sophisticated security beyond traditional password-based authentication. This system analyzes keystroke dynamics and mouse behavior to detect unauthorized access in real-time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture & Components](#architecture--components)
- [Machine Learning Models](#machine-learning-models)
- [Security Features](#security-features)
- [Database Schema](#database-schema)
- [Development & Testing](#development--testing)
- [API Endpoints](#api-endpoints)

## Overview

The Behavioral Authentication System provides multi-layered security through continuous user verification based on behavioral patterns. Rather than relying solely on passwords, this system learns and monitors unique user behavior patterns including keystroke dynamics (timing, rhythm, pressure) and mouse behavior (velocity, acceleration, movement patterns) to authenticate users continuously during their session.

**Key Innovation**: Real-time behavioral anomaly detection combined with drift detection ensures that unauthorized users and account hijacking attempts are identified and blocked immediately.

## Features

### Core Authentication Features
- **Behavioral Biometric Analysis**: Captures and analyzes keystroke and mouse behavior patterns
- **Continuous Authentication**: Real-time verification during active sessions, not just at login
- **Multi-Model Ensemble**: Combines GRU neural networks, autoencoders, SVM, and Isolation Forest for robust detection
- **Drift Detection**: Monitors behavioral changes over time to adapt to natural user evolution
- **Session Management**: Advanced session tracking with comprehensive activity monitoring

### Security Features
- **Intruder Detection**: Automatic workstation lockdown when anomalous behavior is detected
- **JWT Authentication**: Secure token-based API authentication
- **Password Security**: BCrypt hashing with configurable rounds
- **Rate Limiting**: Protection against brute force attacks
- **Comprehensive Logging**: Detailed security event tracking and audit trails

### User Features
- **User Registration**: Account creation with behavioral calibration
- **Calibration Phase**: 30+ second sessions to establish baseline behavioral patterns
- **Challenge Phase**: Verification through additional behavioral samples
- **Responsive UI**: Modern, intuitive interface with real-time feedback

## Technical Stack

### Backend
- **Framework**: Flask with SocketIO for real-time WebSocket communication
- **Authentication**: Flask-JWT-Extended for token-based auth
- **Database**: SQLite3 with comprehensive schema
- **ML/AI**: 
  - TensorFlow/Keras (GRU networks, Autoencoders)
  - scikit-learn (SVM, Isolation Forest, NearestNeighbors)
  - NumPy, Pandas, SciPy for numerical computing

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Real-time data collection and WebSocket communication
- **Styling**: CSS-in-JS with animations and floating elements

### Deployment & Tools
- **Threading**: Multi-threaded asynchronous task handling
- **Logging**: Structured logging with file persistence
- **Version Control**: Python environment management

## Project Structure

```
Behavior_based_Auth/
├── app.py                          # Main Flask application with route handlers
├── config.py                       # Configuration management (dev/prod/test)
├── test_training.py               # Training validation and model testing
├── security_log.txt               # Security event logs
│
├── database/
│   └── db_manager.py             # Database operations and management
│
├── models/
│   ├── behavioral_models.py       # ML model implementations (GRU, Autoencoder, Ensemble)
│   └── saved/                     # Trained model storage
│       └── {user_id}/
│           └── model_*.h5         # Saved Keras models
│
├── utils/
│   ├── feature_extractor.py      # Behavioral feature extraction (38 total features)
│   └── drift_detector.py         # Behavioral drift detection and monitoring
│
├── templates/
│   ├── login.html                # Login interface
│   ├── calib.html                # Calibration/training interface
│   └── challenge.html            # Challenge verification interface
│
└── static/
    ├── css/
    │   └── styles.css            # Application styling
    └── js/
        ├── login.js              # Login form handling
        ├── calib.js              # Calibration data collection
        └── challenge.js          # Challenge verification
```

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Modern web browser with JavaScript enabled

### Steps

1. **Clone the repository**
```bash
cd Behavior_based_Auth
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize the database**
```bash
python -c "from app import create_app; app = create_app(); print('Database initialized')"
```

5. **Run the application**
```bash
python -m flask run
# Or with SocketIO support:
python app.py
```

6. **Access the application**
Open your browser and navigate to `http://localhost:5000`

## Configuration

Edit `config.py` to customize system behavior:

### Key Configuration Parameters

```python
# Behavioral Analysis
WINDOW_SIZE = 30                    # Analysis window in seconds
MIN_CALIBRATION_TIME = 300          # Minimum calibration duration (5 minutes)
KEYSTROKE_FEATURES = [6 features]   # Keystroke dynamics to monitor
MOUSE_FEATURES = [7 features]       # Mouse behavior to monitor

# ML Models
GRU_SEQUENCE_LENGTH = 50            # RNN sequence length
GRU_HIDDEN_UNITS = 64               # GRU layer units
AUTOENCODER_ENCODING_DIM = 32       # Autoencoder bottleneck
ANOMALY_THRESHOLD = 0.15            # Anomaly detection sensitivity

# Authentication Thresholds
CONFIDENCE_THRESHOLD = 0.7          # Minimum confidence for acceptance
ANOMALY_SCORE_THRESHOLD = 0.8       # Maximum acceptable anomaly score
CONSECUTIVE_ANOMALIES_LIMIT = 3     # Lockdown trigger threshold

# Security
BCRYPT_LOG_ROUNDS = 12              # Password hashing strength
SESSION_TIMEOUT = timedelta(hours=8)  # Session expiration
MAX_LOGIN_ATTEMPTS = 5              # Failed login limit
LOCKOUT_DURATION = timedelta(minutes=15)  # Account lockout time

# Drift Detection
DRIFT_ALPHA = 0.05                  # Statistical significance level
DRIFT_MIN_SAMPLES = 30              # Minimum samples for drift detection
BEHAVIORAL_CHANGE_THRESHOLD = 0.25  # Allowable behavior variation
```

### Environment Variables
- `SECRET_KEY`: Flask session secret (set in production)
- `JWT_SECRET_KEY`: JWT signing secret (set in production)
- `DEBUG`: Enable debug mode (default: True)
- `DATABASE_PATH`: Custom database location
- `MODELS_PATH`: Custom models directory

## Usage

### User Registration & Initial Setup

1. **Navigate to Login Page**
   - Access `http://localhost:5000/login`

2. **Create Account**
   - Enter username and password
   - System validates credentials and prompts calibration

3. **Calibration Phase**
   - Navigate to calibration interface
   - Perform typing and mouse movements naturally
   - Minimum 5 minutes of activity required
   - System collects behavioral features:
     - **Keystroke** (6 features): Hold time, flight time, typing speed, rhythm, timing patterns
     - **Mouse** (7 features): Velocity, acceleration, curvature, click patterns, movement direction

4. **Model Training**
   - System trains ensemble of 4 models on calibration data:
     - GRU (Gated Recurrent Unit): Sequential pattern recognition
     - Autoencoder: Anomaly detection
     - One-Class SVM: Outlier identification
     - Isolation Forest: Statistical anomalies

### Challenge & Continuous Authentication

1. **Challenge Phase**
   - Additional verification through behavioral sampling
   - Shorter duration than calibration (quick verification)

2. **Active Session Monitoring**
   - System continuously analyzes keystroke/mouse behavior
   - Real-time confidence scoring
   - Drift detection monitors behavior changes

3. **Anomaly Response**
   - Minor anomalies: Increased monitoring
   - Consistent anomalies (3+): Automatic workstation lockdown
   - Session termination and security logging

## Architecture & Components

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser                             │
│  (login.html, calib.html, challenge.html)                   │
└─────────────┬───────────────────────────────────────────────┘
              │ HTTP/WebSocket
┌─────────────▼───────────────────────────────────────────────┐
│                     Flask Application                        │
│  ├─ Route Handlers (Registration, Login, Calibration)       │
│  ├─ WebSocket Handlers (Real-time data streaming)           │
│  └─ Session Management                                      │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴──────────────────────────┬──────────┐
    │                                    │          │
┌───▼───────────────────┐  ┌────────────▼──────┐  ┌▼──────────────┐
│  Feature Extraction   │  │ ML Model Ensemble │  │ Drift Detector│
│ ─────────────────────  │  │ ────────────────── │  │────────────── │
│ • Keystroke Analysis  │  │ • GRU              │  │ • Statistical │
│ • Mouse Analysis      │  │ • Autoencoder      │  │   Tests       │
│ • Behavioral Metrics  │  │ • One-Class SVM    │  │ • Adaptation  │
│                       │  │ • Isolation Forest │  │   Tracking    │
└───────────────────────┘  └────────────────────┘  └───────────────┘
    │
    └─────────────┬──────────────────────────┐
                  │                          │
        ┌─────────▼─────────┐    ┌──────────▼──────────┐
        │   SQLite Database │    │  Logging & Audit    │
        │ ──────────────────│    │ ──────────────────  │
        │ • Users           │    │ • Security Events   │
        │ • Sessions        │    │ • Behavioral Data   │
        │ • Behavior Data   │    │ • Access Logs       │
        │ • Auth Events     │    │                     │
        │ • Model Metadata  │    │                     │
        └───────────────────┘    └─────────────────────┘
```

### Key Components

#### BehavioralFeatureExtractor
Extracts behavioral features from raw input:

**Keystroke Features (6)**:
- Key hold time (dwell time)
- Flight time (interval between keys)
- Typing speed (WPM - words per minute)
- Pause variance
- Digraph timing
- Trigraph timing

**Mouse Features (7)**:
- Velocity
- Acceleration
- Jerk (rate of acceleration change)
- Curvature
- Click duration
- Dwell time
- Direction changes

#### EnsembleBehavioralClassifier
Combines multiple ML models for robust detection:

1. **GRU Recurrent Network**: Captures sequential behavioral patterns
2. **Autoencoder**: Identifies anomalous feature combinations
3. **One-Class SVM**: Detects outliers in feature space
4. **Isolation Forest**: Statistical anomaly detection

Each model votes on authenticity; ensemble decision improves accuracy and reduces false positives.

#### BehavioralDriftDetector
Monitors behavioral evolution over time:

- **Reference Baseline**: Established during calibration
- **Sliding Windows**: Tracks keystroke and mouse features
- **Statistical Tests**: Uses Kolmogorov-Smirnov and Welch's t-test
- **Feature Weights**: Prioritizes important behavioral indicators
- **Drift Scoring**: Quantifies behavioral deviation

## Machine Learning Models

### Model Training Pipeline

```
Calibration Data (5+ minutes)
        │
        ▼
Feature Extraction (13 features)
        │
        ▼
Feature Normalization & Scaling
        │
        ▼
Model Training
├─ GRU Sequential Model
├─ Autoencoder
├─ One-Class SVM
└─ Isolation Forest
        │
        ▼
Model Persistence (HDF5 format)
        │
        ▼
Production Inference
```

### Model Specifications

**GRU Network**:
- Input: 50-sample sequences of behavioral features
- Architecture: 2 GRU layers (64 → 32 units) with dropout (0.2-0.3)
- Output: Binary classification (genuine/imposter)
- Loss: Binary crossentropy
- Optimizer: Adam (learning rate 0.001)

**Autoencoder**:
- Encoder: Compresses features to 32-dimensional representation
- Decoder: Reconstructs original features
- Anomaly Score: Reconstruction error threshold
- Training: Unsupervised on genuine data only

**One-Class SVM**:
- Kernel: RBF
- Nu Parameter: 0.05 (5% outlier expectation)
- Feature Space: Behavioral vector
- Decision: Distance from learned boundary

**Isolation Forest**:
- Estimators: 100 trees
- Contamination: 0.05 (5% anomaly rate)
- Random State: Reproducible results
- Anomaly Score: Path length in isolation trees

### Prediction Pipeline

```
New Behavioral Sample (keystroke/mouse event)
        │
        ▼
Feature Extraction
        │
        ▼
Ensemble Prediction
├─ GRU prediction (0-1 probability)
├─ Autoencoder anomaly score
├─ One-Class SVM decision
└─ Isolation Forest score
        │
        ▼
Weighted Voting
        │
        ▼
Confidence Score (0-1)
        │
        ▼
Decision: Accept/Reject/Investigate
```

## Security Features

### Authentication & Authorization
- **JWT Tokens**: All API requests require valid JWT
- **Session Management**: Active session tracking with automatic timeout
- **Rate Limiting**: Brute force protection on login endpoint
- **Password Security**: BCrypt with configurable rounds (12-14 for production)

### Continuous Verification
- **Real-time Anomaly Detection**: Monitors every keystroke and mouse movement
- **Ensemble Voting**: Multiple models reduce false positives
- **Adaptive Thresholds**: Production mode (confidence: 0.8, anomaly: 0.75) tighter than development
- **Confidence Boosting**: Higher confidence (lower threshold) early in session

### Threat Response
- **Automatic Lockdown**: Workstation locked after 3 consecutive anomalies (2 in production mode)
- **10-Second Delay**: Prevents rapid repeated access attempts
- **Cross-Platform**: Linux, macOS, and Windows support
- **Security Logging**: Complete audit trail of all events

### Data Protection
- **Behavioral Data Isolation**: Per-user models prevent cross-user attacks
- **Feature Normalization**: Prevents data leakage between models
- **Secure Session Storage**: SQLite with proper indexing
- **Audit Logging**: Comprehensive event tracking with timestamps

## Database Schema

### users Table
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    salt TEXT,
    created_at TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN,
    failed_attempts INTEGER,
    locked_until TIMESTAMP,
    calibration_complete BOOLEAN
)
```

### sessions Table
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER,
    created_at TIMESTAMP,
    last_activity TIMESTAMP,
    is_active BOOLEAN,
    ip_address TEXT,
    user_agent TEXT
)
```

### behavioral_data Table
```sql
CREATE TABLE behavioral_data (
    data_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    session_id TEXT,
    timestamp TIMESTAMP,
    data_type TEXT,           -- 'keystroke' or 'mouse'
    features TEXT,            -- JSON of extracted features
    raw_data TEXT,           -- Raw measurements
    confidence_score REAL,
    anomaly_score REAL
)
```

### auth_events Table
```sql
CREATE TABLE auth_events (
    event_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    session_id TEXT,
    event_type TEXT,         -- 'login', 'logout', 'anomaly', 'drift'
    event_data TEXT,         -- JSON event details
    timestamp TIMESTAMP,
    ip_address TEXT
)
```

### model_metadata Table
```sql
CREATE TABLE model_metadata (
    user_id INTEGER PRIMARY KEY,
    model_version INTEGER,
    last_trained TIMESTAMP,
    training_samples INTEGER,
    model_accuracy REAL,
    drift_detected BOOLEAN,
    drift_timestamp TIMESTAMP
)
```

## API Endpoints

### Authentication Endpoints

**POST /api/register**
- Register new user
- Body: `{username, email, password}`
- Returns: `{user_id, token, redirect_url}`

**POST /api/login**
- User login
- Body: `{username, password}`
- Returns: `{session_id, token, redirect_url}`

**POST /api/logout**
- End session
- Returns: `{status, message}`

### Calibration Endpoints

**POST /api/calibration/start**
- Initiate calibration phase
- Body: `{session_id}`
- Returns: `{calibration_id, duration_required}`

**POST /api/calibration/submit**
- Submit calibration data
- Body: `{calibration_id, keystroke_data, mouse_data}`
- Returns: `{status, training_result}`

### Challenge Endpoints

**POST /api/challenge/start**
- Initiate challenge verification
- Body: `{session_id}`
- Returns: `{challenge_id, duration}`

**POST /api/challenge/verify**
- Submit challenge data for verification
- Body: `{challenge_id, keystroke_data, mouse_data}`
- Returns: `{status, confidence_score, authenticated}`

### Behavioral Monitoring (WebSocket)

**Real-time Data Streaming**
- Event: `keystroke_event` → `{key, timestamp, hold_time, flight_time}`
- Event: `mouse_event` → `{x, y, timestamp, event_type, speed}`
- Event: `anomaly_detected` → `{user_id, anomaly_score, recommendation}`

## Development & Testing

### Running Tests

```bash
# Run model training validation
python test_training.py

# Expected output:
# Dims: ks=6 ms=7 combined=13
# Created 100 genuine feature samples
# TRAINING SUCCESS — dimensions are correct
# PREDICTION SUCCESS — result: [score]
# SYSTEM READY — safe to calibrate
```

### Development Mode

```bash
# Enable debug mode
export DEBUG=True
python app.py
```

### Monitoring & Logging

Security events logged to `behavioral_auth.log`:
```
2024-01-15 10:30:45 - app - INFO - User registered: john_doe
2024-01-15 10:35:20 - app - INFO - Calibration complete for user 1
2024-01-15 10:40:15 - app - ERROR - INTRUDER DETECTED - User: 1 - Score: 0.87
2024-01-15 10:40:25 - app - INFO - Workstation locked for user 1
```

Also see `security_log.txt` for authentication events:
```
[2024-01-15_10-40-15] INTRUDER DETECTED - User: 1 - Score: 0.87
```

### Troubleshooting

**Database errors**: Ensure `database/` directory exists and is writable
```bash
mkdir -p database
```

**Model loading issues**: Verify model files exist in correct path
```bash
ls models/saved/{user_id}/
```

**WebSocket connection failures**: Check browser console and CORS settings in config.py

**Feature dimension mismatches**: Run `test_training.py` to validate setup

## Performance Metrics

- **Feature Extraction**: ~5-10ms per keystroke/mouse event
- **Ensemble Prediction**: ~50-100ms per sample
- **Session Overhead**: <2% CPU for active monitoring
- **Memory Usage**: ~100-200MB per active user model

## Future Enhancements

- Multi-factor behavioral patterns (gait recognition via touchpad motion)
- Adaptive thresholding based on user activity patterns
- Cross-device behavioral consistency verification
- Integration with other authentication methods (FIDO2, WebAuthn)
- Machine learning model retraining automation
- Enhanced visualization dashboard for security monitoring
- Performance optimization for large-scale deployments

## Security Considerations

1. **Production Deployment**: Change all SECRET_KEY and JWT_SECRET_KEY values
2. **HTTPS Only**: Always use HTTPS in production
3. **Database Encryption**: Consider encrypting behavioral_data table
4. **User Privacy**: Behavioral data should not be shared externally
5. **Model Security**: Protect trained models from tampering
6. **Regular Testing**: Perform penetration testing and anomaly injection tests
7. **Log Rotation**: Implement log rotation for security audit trails

## License

This project is provided as-is for educational and security research purposes.

## Contributors

Developed as an advanced behavioral biometric authentication system demonstrating machine learning applications in cybersecurity.

---

**Last Updated**: April 2026  
**Version**: 1.0.0

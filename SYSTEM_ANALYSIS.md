# Behavior-Based Continuous Authentication System - Complete Analysis

## EXECUTIVE SUMMARY

This is an advanced **real-time behavioral biometrics authentication system** that continuously verifies user identity throughout their active session by analyzing keystroke dynamics and mouse behavior patterns. It uses machine learning ensemble models to detect anomalies and potential intruders, with automatic workstation lockdown capabilities.

---

## 1. PROJECT OVERVIEW

### Purpose
- **Primary Goal**: Provide continuous, transparent authentication after initial login
- **Security Method**: Behavioral biometrics (keystroke dynamics + mouse patterns)
- **Status**: In active development with multiple phases completed

### Key Statistics
- **Total Features Extracted**: 38 (18 keystroke + 20 mouse)
- **Feature Dimensions**: 38-dimensional feature vector
- **Minimum Calibration Time**: 30 seconds
- **Window Size**: 100 keystrokes for rolling window classification
- **Step Size**: 25 new keystrokes before performing authentication

---

## 2. SYSTEM ARCHITECTURE

### Folder Structure
```
Behavior_based_Auth/
├── app.py                          # Flask main app + SocketIO handlers
├── config.py                       # Configuration & thresholds
├── requirements.txt                # Python dependencies
├── security_log.txt                # Lockdown events log
├── test_training.py                # Dimension verification tests
│
├── models/
│   ├── behavioral_models.py        # Ensemble classifier (6 models)
│   └── saved/
│       ├── 1/, 2/, ...             # Per-user model directories
│       │   ├── model_gru.h5        # TensorFlow GRU model
│       │   ├── model_autoencoder.h5
│       │   └── sklearn_models.pkl  # One-Class SVM, Isolation Forest, etc.
│
├── utils/
│   ├── feature_extractor.py        # 38-feature extraction pipeline
│   ├── drift_detector.py           # Behavioral drift analysis
│   └── __pycache__/
│
├── database/
│   ├── db_manager.py               # SQLite database layer
│   ├── auth_system.db              # SQLite database file
│   └── __pycache__/
│
├── static/
│   ├── css/
│   │   └── styles.css              # Dark theme styling
│   └── js/
│       ├── login.js                # Registration/Login UI
│       ├── calib.js                # Calibration data collection
│       └── challenge.js            # Real-time monitoring dashboard
│
└── templates/
    ├── login.html                  # Login/Register page
    ├── calib.html                  # Calibration page
    └── challenge.html              # Secure dashboard
```

### High-Level Data Flow

```
User Types → Frontend JS (calib.js/challenge.js)
    ↓
Keystroke/Mouse Events Captured
    ↓
Behavioral Data Buffer (rolling window)
    ↓
Feature Extraction (38 dimensions)
    ↓
ML Ensemble Models (6 models vote)
    ↓
Authentication Score + Anomaly Detection
    ↓
Alert/Lockdown Decision
    ↓
Database Logging + WebSocket Response
```

---

## 3. ML ENSEMBLE ARCHITECTURE

### Six Models in Ensemble Classifier

1. **GRU (Gated Recurrent Unit)**
   - TensorFlow/Keras sequential model
   - Architecture: 2-layer GRU with 64→32 hidden units
   - Input shape: (sequence_length=50, features=38)
   - Handles temporal dependencies in keystroke patterns
   - File: `models/saved/{user_id}/model_gru.h5`

2. **Autoencoder**
   - Input: 38 features
   - Encoding dimension: 32
   - Detects anomalies via reconstruction loss
   - Trains to reconstruct normal behavior
   - Anomaly = High reconstruction error

3. **One-Class SVM**
   - Scikit-learn model
   - Trains on normal user behavior
   - Returns distance from decision boundary
   - Higher distance = more anomalous

4. **Isolation Forest**
   - Scikit-learn ensemble
   - Anomaly score based on isolation depth
   - No assumptions about data distribution
   - Effective for rare behavioral patterns

5. **Passive-Aggressive Classifier**
   - Online learning classifier
   - Updates incrementally with each new feature
   - Binary classification (genuine vs. intruder)
   - Adapts to gradual behavior changes

6. **k-Nearest Neighbors (Incremental)**
   - Nearest Neighbors with sliding window
   - Compares new features against recent history
   - Distance-based anomaly detection
   - Remembers last N samples

### Ensemble Voting Strategy
- Each model outputs an anomaly score (0-1)
- Models vote on whether sample is "genuine" or "intruder"
- **Consensus Score** = average of all 6 models
- **Authenticity Score** = 1 - Consensus Anomaly Score
- **Alert Level** determined by: authenticity_score + model_confidence

---

## 4. FEATURE EXTRACTION PIPELINE

### 38 Total Features

#### Keystroke Dynamics (18 features)
```
Hold Times:
  - hold_time_mean        (average key press duration, ms)
  - hold_time_std         (variability in key press duration)
  - hold_time_median      (middle value, resistant to outliers)

Flight Times (time between key releases):
  - flight_time_mean      (average inter-keystroke interval, ms)
  - flight_time_std       (variability between keystrokes)
  - flight_time_median    (middle value)

Typing Speed:
  - typing_speed_wpm      (words per minute)
  - typing_speed_cpm      (characters per minute)

Rhythm & Patterns:
  - rhythm_consistency    (how consistent typing pace is 0-1)
  - burst_ratio           (fast typing percentage)
  - pause_ratio           (slow/paused typing percentage)
  - avg_pause_duration    (average pause length, ms)
  - speed_variance        (measure of speed fluctuation)
  - speed_trend           (acceleration/deceleration tendency)

Consistency Measures:
  - digraph_consistency   (consistency of 2-key sequences)
  - hold_time_cv          (coefficient of variation: std/mean)
  - flight_time_cv        (coefficient of variation: std/mean)
  - pressure_consistency  (how uniform keystroke force is)
```

#### Mouse Dynamics (20 features)
```
Velocity Metrics:
  - velocity_mean         (average movement speed, pixels/ms)
  - velocity_std          (speed variability)
  - velocity_median       (middle speed value)

Acceleration:
  - acceleration_mean     (average change in speed)
  - acceleration_std      (variability in acceleration)

Movement Patterns:
  - movement_efficiency   (directness of movement 0-1)
  - curvature_mean        (average bend in path)
  - curvature_std         (bend variability)
  - avg_direction_change  (average angle change between segments)
  - direction_change_variance

Click Behavior:
  - click_duration_mean   (average mouse button hold time, ms)
  - click_duration_std    (click duration variability)
  - left_click_ratio      (percentage left clicks)
  - right_click_ratio     (percentage right clicks)

Inter-Click & Dwell:
  - inter_click_mean      (average time between clicks, ms)
  - inter_click_std       (click interval variability)
  - dwell_time_mean       (average stationary hover time, ms)

Movement Geometry:
  - movement_area         (bounding box area of mouse movement)
  - movement_centrality   (how centered movement is around click points)
  - velocity_smoothness   (smoothness of velocity curve 0-1)
```

### Feature Extraction Process

1. **Data Collection Phase**
   ```
   Keystroke Event: {
     "key": "a",
     "press_time": 1234567890,
     "dwell_time": 85,        // key held for 85ms
     "flight_time": 150       // time since last key released
   }
   
   Mouse Event: {
     "type": "mousemove|mousedown|mouseup|click",
     "x": 512,
     "y": 384,
     "timestamp": 1234567890,
     "button": 0  // 0=left, 1=middle, 2=right
   }
   ```

2. **Aggregation Phase**
   - For keystrokes: group 20-100 events into one feature extraction
   - Calculate statistical measures (mean, std, median, etc.)
   - For mouse: calculate velocities, accelerations, distances

3. **Normalization Phase**
   - Fixed feature order: keystroke features first (18), then mouse (20)
   - Missing values filled with 0
   - All features guaranteed to output exactly 38 values
   - StandardScaler applied before ML models

---

## 5. AUTHENTICATION FLOW

### User Registration/Login
1. User registers with username, email, password
2. Password hashed using bcrypt (log rounds: 12)
3. User record created in SQLite database
4. User redirected to login page

### Calibration Phase (30+ seconds)
1. User logs in
2. Redirected to `/calibration` page
3. User types text + moves mouse for at least 30 seconds
4. Behavioral data sent every 3 seconds via SocketIO to `/behavioral_data`
5. Backend stores 20+ keystroke samples + 20+ mouse samples
6. At calibration completion:
   - Features extracted from all collected data
   - 6 ML models trained on user's baseline behavior
   - Models saved to `models/saved/{user_id}/`
   - Drift detector baseline established
   - `calibration_complete` flag set to True in database

### Real-Time Authentication (Continuous)
1. User navigates to `/challenge` page
2. Establishes WebSocket connection via SocketIO
3. `join_session` emitted with session_id
4. Keystroke/mouse data continuously captured in background
5. Every 3 seconds or after 25 keystrokes:
   - Feature extraction performed on rolling window
   - 6 models predict authenticity
   - Consensus score calculated
   - Anomaly score = 1 - authenticity_score
   - Decision logic applied:

```
DECISION LOGIC:
  IF anomaly_score > ANOMALY_SCORE_THRESHOLD (0.36):
    IF model_confidence > 0.7:
      Alert Level 3 (Critical) → Request Immediate Re-auth
    ELIF model_confidence > 0.5:
      Alert Level 2 (Warning) → Monitor
    ELSE:
      Alert Level 1 (Info) → Continue monitoring
  
  IF drift_detected (behavioral pattern changed):
    IF alert_level == 0:
      Alert Level 1 → Notify user
  
  CONSECUTIVE ANOMALIES >= 3:
    Cooldown check: if last lockdown was 30+ seconds ago
      TRIGGER LOCKDOWN (Windows: rundll32.exe user32.dll,LockWorkStation)
      Send 10-second countdown to frontend alert overlay
```

---

## 6. DATA STRUCTURES

### Database Schema

#### Users Table
```sql
user_id             INTEGER PRIMARY KEY
username            TEXT UNIQUE
email               TEXT UNIQUE
password_hash       TEXT
salt                TEXT
created_at          TIMESTAMP
last_login          TIMESTAMP
is_active           BOOLEAN (default: 1)
failed_attempts     INTEGER (default: 0)
locked_until        TIMESTAMP
calibration_complete BOOLEAN (default: 0)
```

#### Sessions Table
```sql
session_id          TEXT PRIMARY KEY (UUID-like)
user_id             INTEGER FOREIGN KEY
created_at          TIMESTAMP
last_activity       TIMESTAMP
is_active           BOOLEAN (default: 1)
ip_address          TEXT
user_agent          TEXT
```

#### Behavioral Data Table
```sql
data_id             INTEGER PRIMARY KEY
user_id             INTEGER FOREIGN KEY
session_id          TEXT FOREIGN KEY
timestamp           TIMESTAMP
data_type           TEXT ('keystroke' or 'mouse')
features            TEXT (JSON: 38 feature values)
raw_data            TEXT (JSON: raw keystroke/mouse events)
confidence_score    REAL
anomaly_score       REAL
```

#### Auth Events Table
```sql
event_id            INTEGER PRIMARY KEY
user_id             INTEGER FOREIGN KEY
session_id          TEXT FOREIGN KEY
event_type          TEXT ('login', 'logout', 'anomaly', 'drift')
event_data          TEXT (JSON: event details)
timestamp           TIMESTAMP
ip_address          TEXT
```

#### Model Metadata Table
```sql
user_id             INTEGER PRIMARY KEY
model_version       INTEGER (default: 1)
last_trained        TIMESTAMP
training_samples    INTEGER
model_accuracy      REAL
drift_detected      BOOLEAN (default: 0)
drift_timestamp     TIMESTAMP
```

---

## 7. CONFIGURATION THRESHOLDS (config.py)

### ML Model Configuration
```python
# Window for feature extraction
WINDOW_SIZE = 10  # seconds

# Calibration requirements
MIN_CALIBRATION_TIME = 30  # seconds
KEYSTROKE_BUFFER_SIZE = 1000
MOUSE_BUFFER_SIZE = 2000

# GRU Model
GRU_SEQUENCE_LENGTH = 10
GRU_HIDDEN_UNITS = 64

# Autoencoder
AUTOENCODER_ENCODING_DIM = 32
ANOMALY_THRESHOLD = 0.15

# Drift Detection
DRIFT_DETECTION_WINDOW = 20
```

### Authentication Thresholds
```python
CONFIDENCE_THRESHOLD = 0.6
ANOMALY_SCORE_THRESHOLD = 0.36  # score > 0.36 = anomaly
CONSECUTIVE_ANOMALIES_LIMIT = 3  # 3 strikes → lockdown
```

### Security Configuration
```python
BCRYPT_LOG_ROUNDS = 12
SESSION_TIMEOUT = 8 hours
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 15 minutes
```

---

## 8. FRONTEND JAVASCRIPT ARCHITECTURE

### login.js
**Purpose**: User registration and login interface

**Key Components**:
- LoginManager class handles form submissions
- Password strength validation (8+ chars, uppercase, lowercase, numbers, symbols)
- Form switching (login ↔ register)
- API calls to `/api/register` and `/api/login`
- Stores session_id, user_id, username in localStorage

**Data Flow**:
```javascript
handleLogin() → 
  POST /api/login {username, password} → 
  Receive {access_token, session_id, user_id, calibration_complete} →
  Store in localStorage →
  Redirect to calibration or challenge
```

### calib.js
**Purpose**: Behavioral baseline calibration (30+ seconds)

**Key Components**:
- CalibrationManager class
- Real-time keystroke/mouse event capturing
- 30-second countdown timer
- Progress bar showing data collection
- SocketIO connection for real-time data sending

**Data Capture**:
```javascript
// Keystroke events
document.addEventListener('keydown', (e) => {
  keystrokeBuffer.push({
    key: e.key,
    press_time: Date.now(),
    flight_time: millisSinceLastKey
  });
});

document.addEventListener('keyup', (e) => {
  lastKeystroke.dwell_time = Date.now() - press_time;
});

// Mouse events
document.addEventListener('mousemove', (e) => {
  mouseEvents.push({
    type: 'mousemove',
    x: e.clientX,
    y: e.clientY,
    timestamp: Date.now()
  });
});

// Send every 3 seconds
setInterval(() => {
  socket.emit('behavioral_data_calibration', {
    keystroke_data: keystrokeBuffer.splice(0),
    mouse_data: mouseEvents.splice(0)
  });
}, 3000);
```

### challenge.js
**Purpose**: Real-time behavioral monitoring dashboard

**Key Classes**:
- Keystroke capture at lines 51-69 (global keystrokeBuffer)
- Mouse event capture via document listeners
- DashboardManager class (starts line 240) - manages UI/charts
- Real-time authentication via SocketIO

**Current Issue** (CRITICAL BUG):
```javascript
// LINES 51-69: Global keystroke buffer collecting data
const keystrokeBuffer = [];
document.addEventListener('keydown', (e) => {
  keystrokeBuffer.push({...});  // ✅ Adds to global keystrokeBuffer
});

// LINES 180-230: sendBehavioralDataForAuth function
function sendBehavioralDataForAuth() {
  const keystrokeData = [...keystrokeBuffer];  // ✅ Reads from global
  keystrokeBuffer.length = 0;
}

// BUT ALSO LINES 1517-1524: Duplicate keystroke listeners inside separate DOMContentLoaded
document.addEventListener('keydown', (e) => {
  keystrokeEvents.push({...});  // ⚠️ Adds to DIFFERENT keystrokeEvents array
});

// RESULT: 
// - keystrokeBuffer is filled ✅
// - keystrokeEvents is filled ✅
// - sendBehavioralDataForAuth reads keystrokeBuffer ✅
// - BUT: Multiple DOMContentLoaded handlers and duplicate listeners 
//   cause event handler chain confusion
```

**Current Symptoms**:
- Backend receives behavioral_data events
- keystroke_data array is EMPTY `[]`
- Terminal prints: `[BUFFER] 0 total keystrokes`
- No authentication scoring occurs
- Console logs missing keystroke counts

**Root Cause Analysis**:
The keystrokeBuffer IS being populated (by lines 51-69), but:
1. Triple keystroke capturing happening (lines 51-69, 1517-1524, DashboardManager.captureKeystroke)
2. Multiple `DOMContentLoaded` event handlers (lines 77, 1433, 1469)
3. Unclear which event listener actually gets attached to document
4. Race condition between sendBehavioralDataForAuth (inside DOMContentLoaded scope) and global keystrokeBuffer (outside)

---

## 9. CURRENT ISSUES

### CRITICAL ISSUE #1: Keystroke Buffer Not Reaching Backend ⚠️⚠️⚠️

**Status**: JUST DIAGNOSED AND DEBUG LOGS ADDED

**Symptom**:
```
Terminal Output:
[BUFFER] 0 total keystrokes, 0 since last classification
[FILLING] 0/100 keystrokes collected

Browser Console (Missing):
Keystroke recorded, buffer size: 1
Keystroke recorded, buffer size: 2
...
Sending buffer size: 5
```

**Exact Problem**:
- User types but `keystrokeBuffer` remains empty in backend
- Debug logs added (March 2024) show buffer is never filled
- Issue is in `challenge.js` keystroke listener chain

**What We Just Added**:
- Line 60 in challenge.js: `console.log('Keystroke recorded, buffer size:', keystrokeBuffer.length);`
- Line 188 in challenge.js: `console.log('Sending buffer size:', keystrokeBuffer.length);`

**Next Steps to Confirm Fix**:
1. Hard refresh with Ctrl+Shift+R
2. Open browser console (F12)
3. Type 5 keys
4. Check console for logs
5. If showing `buffer size: 0`, then multiple competing listeners are the issue

**Proposed Permanent Fix**:
- Remove duplicate keystroke listeners (lines 1517-1524)
- Remove duplicate DOMContentLoaded handler (lines 1469+)
- Keep only lines 51-69 (global keystrokeBuffer)
- Ensure sendBehavioralDataForAuth accesses correct buffer

---

### ISSUE #2: Multiple DOMContentLoaded Handlers

**Location**: challenge.js has 3 DOMContentLoaded handlers
- Handler 1: Line 77 (initializes auth flow)
- Handler 2: Line 1433 (DashboardManager initialization)
- Handler 3: Line 1469 (Socket.io-based behavioral_data sender)

**Problem**: Unclear execution order, may interfere with event listeners

**Solution**: Consolidate to single DOMContentLoaded with all initialization

---

### ISSUE #3: Feature Dimension Consistency (PARTIALLY FIXED)

**Status**: v2.1.0 updates helped, but lingering issues

**What Was Fixed**:
- Feature count verified as 38 (18 keystroke + 20 mouse)
- Dimension normalization functions added
- `ensure_dim()` helper in app.py handles padding/trimming

**Remaining Issues**:
- Some feature extraction paths may return different shapes
- Edge cases with empty keystroke data
- Synthetic feature generation fallback may not match real features

**Mitigations**:
- `fix_feature_dimensions()` in BehavioralFeatureExtractor
- Padding with zeros for missing features
- Trimming excess features to match model expectations

---

### ISSUE #4: Drift Detection Configuration Missing

**Status**: Partially implemented

**Problem**:
- `config.py` references `DRIFT_ALPHA` and `DRIFT_MIN_SAMPLES`
- These values not actually defined in config.py
- BehavioralDriftDetector may fail during initialization

**Affected Code**:
- app.py line ~165: `alpha=app.config['DRIFT_ALPHA']`
- app.py line ~165: `min_samples=app.config['DRIFT_MIN_SAMPLES']`

**Solution Needed**:
Add to config.py:
```python
DRIFT_ALPHA = 0.1  # smoothing factor for drift detection
DRIFT_MIN_SAMPLES = 10  # minimum samples to detect drift
```

---

### ISSUE #5: Browser Compatibility

**Known Issues**:
- Edge browsers may have different keystroke event handling
- Some VPN/security tools block performance.now() timing
- Virtual keyboard inputs may not trigger proper events

**Workaround**:
- Feature extraction includes fallbacks for missing timing data
- Synthetic data generation when real data insufficient

---

## 10. TYPICAL USER JOURNEY

### Step 1: Registration
```
User → Click "New User?" link
User → Fill: Username, Email, Password (8+ chars)
User → Email already exists OR Username taken → Error shown
User → Click Register → Account created
User → Redirected to Login
```

### Step 2: Login
```
User → Enter username & password
Backend → Verify credentials
Backend → Create session + JWT token
Backend → Return session_id to frontend
Frontend → Store session_id, user_id in localStorage
Frontend → Check calibration_complete flag
  IF false → Redirect to /calibration
  IF true → Redirect to /challenge
```

### Step 3: Calibration (First Time Only)
```
User → Visits /calibration
CalibrationManager → Starts 30-second countdown
CalibrationManager → Listens for keystrokes/mouse
CalibrationManager → Every 3 seconds: emit behavioral_data_calibration
Backend → Store data in behavioral_data table
Backend → UI shows "Collecting behavioral data... 12/30 seconds"
User → Types freely, moves mouse, clicks buttons
At 30 seconds:
  Backend → Extract features (keystroke + mouse)
  Backend → Train 6 ML models
  Backend → Save models to disk
  Backend → Set calibration_complete = True
  Frontend → Show "Calibration complete!" 
  Frontend → Redirect to /challenge
```

### Step 4: Continuous Authentication (Every Session)
```
User → Visits /challenge
DashboardManager → Connect to SocketIO
DashboardManager → Join session with session_id
Every keystroke/mouse move:
  → Captured by event listeners
  → Added to keystrokeBuffer / mouseEvents arrays
Every 3 seconds OR after 25 keystrokes:
  → sendBehavioralDataForAuth() called
  → Emit behavioral_data event with keystrokeBuffer contents
Backend:
  → Receive behavioral_data
  → Add to rolling window buffer (last 100 keystrokes)
  → Extract features (38 dimensions)
  → Run through 6 ML models
  → Calculate consensus anomaly score
  IF anomaly_score > 0.36:
    → Send security_alert to frontend (yellow or red overlay)
  IF 3 consecutive anomalies:
    → Trigger lockdown (if 30+ seconds since last lockdown)
    → Send lockdown_initiated with 10-second countdown
    → Frontend shows red "INTRUDER DETECTED" overlay
    → After 10 seconds: Windows workstation locks
```

---

## 11. DEPLOYMENT & REQUIREMENTS

### Python Version
- Tested: Python 3.8-3.10
- NOT compatible with Python 3.11+ (TensorFlow/Keras compatibility issues)

### Core Dependencies
```
flask==2.3.2              # Web framework
flask-socketio==5.3.4     # Real-time WebSocket
flask-jwt-extended==4.4.4 # JWT token management
tensorflow==2.13.0        # Deep learning (GRU, Autoencoder)
scikit-learn==1.3.0       # Ensemble models (SVM, Random Forest, etc.)
numpy==1.24.3             # Numerical computing
pandas==2.0.3             # Data manipulation
bcrypt==4.0.1             # Password hashing
python-socketio==5.9.0    # SocketIO support
```

### Installation Steps
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Visit http://localhost:5000
```

---

## 12. LOGGING & DEBUGGING

### Log Files
- **behavioral_auth.log**: Main application logs (DEBUG level)
- **security_log.txt**: Lockdown events only

### Debug Output Examples

**Dimension Measurement**:
```
[DIMENSIONS] keystroke=18 mouse=20 combined=38
[DIM FIX] keystroke_features: padding 15 → 38
```

**Calibration**:
```
[TRAINING] keystroke features shape: 25 samples, 18 dims
[TRAINING] mouse features shape: 25 samples, 20 dims
[TRAINING] expected dims: ks=18 ms=20
[TRAINING] Starting model training...
[TRAINING] All models trained successfully
```

**Real-Time Authentication**:
```
[BUFFER] 100 total keystrokes, 25 since last classification
[WINDOW] Classifying on 100 keystrokes
[ANOMALY CHECK] User 1: anomaly_score=0.150, threshold=0.360, confidence=0.823
[AUTHORIZED] score=0.850 conf=1.00 strikes=0/3
```

**Anomaly Detection**:
```
[ANOMALY] score=0.425 conf=0.92 strikes=1/3
[STRIKE] 2/3
[LOCKDOWN TRIGGERED]
[COOLDOWN] 27s remaining
```

---

## 13. SECURITY CONSIDERATIONS

### Password Security
- Bcrypt hashing with 12 log rounds
- Salts included in password_hash
- Passwords minimum 8 characters
- Failed attempts tracked (max 5 before 15-min lockout)

### Session Security
- Session IDs are UUID-like strings
- Sessions stored in database with:
  - User ID mapping
  - IP address logging
  - User-Agent string logging
  - Last activity timestamp
  - is_active flag
- Sessions timeout after 8 hours of inactivity

### Behavioral Data Security
- All keystroke/mouse data stored in SQLite (local)
- No transmission of raw behavioral data to cloud
- Features extracted locally before ML processing
- Models trained on client user's machine only

### WebSocket Security
- CORS enabled for all origins (development mode)
  - Production: Should restrict to specific frontend URL
- Session authentication on every SocketIO event
- Invalid sessions rejected with "session expired" error

---

## 14. TESTING & VALIDATION

### Unit Tests Available
- `test_training.py`: Validates feature dimensions
  - Verifies keystroke features = 18
  - Verifies mouse features = 20
  - Verifies combined = 38
  - Tests feature extraction edge cases

### Manual Testing Checklist
- [ ] User registration (username taken, email taken, valid creation)
- [ ] User login (wrong password, successful login)
- [ ] Calibration phase (types for 30 seconds, models train)
- [ ] Real-time auth (typing collected, buffer fills)
- [ ] Anomaly detection (spoof typing patterns → anomaly alert)
- [ ] Lockdown trigger (3 consecutive anomalies → trigger)
- [ ] Workstation lock (Windows lock command executes)

### Debug Mode Flags
- Set `DEBUG = True` in config
- Prints extensive logging to console
- Synthetic data generation for testing
- Reduced calibration time (30 seconds instead of 5 minutes)

---

## 15. ARCHITECTURE DECISIONS & RATIONALE

### Why Ensemble of 6 Models?
- **Diversity**: Different model types reduce bias
- **Robustness**: One model failure doesn't break system
- **Coverage**: Each handles different anomaly types
  - GRU: Temporal patterns
  - Autoencoder: Reconstruction-based anomalies
  - One-Class SVM: Boundary-based outliers
  - Isolation Forest: Rare patterns
  - Passive-Aggressive: Online learning
  - k-NN: Local neighborhood anomalies

### Why 38 Features?
- **Balance**: 18 keystroke + 20 mouse provides good keystroke/mouse ratio
- **Comprehensive**: Covers timing, speed, consistency, patterns, geometry
- **ML-Friendly**: Not too high-dimensional (curse of dimensionality)
- **Real-Time**: Can compute in <100ms per classification

### Why Rolling Window Instead of Batch?
- **Continuous Auth**: Real-time decisions required
- **Adaptation**: Recent behavior weighted more
- **Efficiency**: Only classify on new data (sliding window)
- **Low Latency**: No waiting for large batch

### Why SocketIO Instead of REST?
- **Bidirectional**: Server can push alerts to client instantly
- **Persistent Connection**: No reconnect overhead
- **Real-Time**: Low latency for security events
- **Stateful**: Session context preserved across messages

---

## 16. FUTURE IMPROVEMENTS

### Planned Features
1. **Gait Recognition**: Incorporate trackpad movement patterns
2. **Pressure Sensitivity**: Analyze key press force (if device supports)
3. **EEG/Heart Rate**: Integration with biometric wearables
4. **Progressive Drift Adaptation**: Slowly adapt models to legitimate behavior changes
5. **Threat Intelligence**: Signature database of known intruder patterns
6. **Multi-Factor Step-Up**: Request additional auth (password, TOTP) on anomaly

### Performance Optimizations
1. **Model Quantization**: Reduce ML model file sizes by 75%
2. **GPU Acceleration**: Use CUDA for GRU inference (100x faster)
3. **Caching Features**: Cache extracted features to avoid recomputation
4. **Async Processing**: Move model training to background tasks

### Security Enhancements
1. **Encrypted Storage**: Encrypt behavioral data at rest
2. **FIPS Compliance**: Use FIPS-approved algorithms only
3. **Audit Trail**: Immutable log of all authentication events
4. **Geofencing**: Reject logins from impossible locations

---

## SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **System Type** | Behavioral Biometrics + ML Ensemble |
| **Total Features** | 38 (18 keystroke + 20 mouse) |
| **ML Models** | 6 (GRU, Autoencoder, One-Class SVM, Isolation Forest, Passive-Aggressive, k-NN) |
| **Min Calibration** | 30 seconds |
| **Reauth Window** | 100 keystrokes |
| **Reauth Trigger** | 25 new keystrokes or 3 seconds |
| **Anomaly Threshold** | 0.36 (anomaly_score) |
| **Lockdown Trigger** | 3 consecutive anomalies |
| **Lockdown Cooldown** | 30 seconds |
| **Database** | SQLite (auth_system.db) |
| **WebSocket** | Flask-SocketIO |
| **Terminal Output** | [BUFFER] [WINDOW] [ANOMALY] [STRIKE] [LOCKDOWN] tags |
| **Current Critical Issue** | Keystroke buffer empty in backend (event listener conflict) |
| **Debug Logs Added** | Lines 60 & 188 in challenge.js |

---

## CONTACT & DOCUMENTATION

- **Repository**: https://github.com/Magizharasi/Behavior_based_Auth
- **Dashboard UI**: `/challenge` route
- **Calibration UI**: `/calib` route  
- **API Base**: `http://localhost:5000`
- **SocketIO Endpoint**: Same base URL with WebSocket upgrade

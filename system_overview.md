# Behavioral Authentication System — System Overview

## 1. High-Level Architecture

This project implements a **continuous authentication system** using behavioral biometrics. Instead of a one-time password check, the system continuously monitors how a user types and moves their mouse, extracting behavioral features and running them through an ensemble of 6 machine learning models to verify the user's identity in real time.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (Frontend)                          │
│  login.js  │  calib.js  │  challenge.js  │  Chart.js (dashboard)   │
└────────────┬───────────────────────────────┬───────────────────────┘
             │ REST API                       │ Socket.IO (WebSocket)
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Flask Server (app.py)                             │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ REST API  │  │ WebSocket  │  │ Auth Logic   │  │ Background   │  │
│  │ Routes    │  │ Events     │  │ (JWT+Session)│  │ Tasks        │  │
│  └──────────┘  └────────────┘  └──────────────┘  └──────────────┘  │
└────────┬──────────────┬──────────────────────┬──────────────────────┘
         │              │                      │
         ▼              ▼                      ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│  SQLite DB   │ │  Features    │ │  6 ML Models (Ensemble)  │
│  (db_manager)│ │  (extractor) │ │  (behavioral_models.py)  │
└──────────────┘ └──────────────┘ └──────────────────────────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │ Drift Detector  │
                              │ (statistical)   │
                              └────────────────┘
```

## 2. Directory Structure

```
Behavior_based_Auth/
├── .env                                              # Environment variables
├── requirements.txt                                  # Python dependencies
├── README.md                                         # Project readme
│
└── Behavior_based_Auth/                              # Main application package
    ├── app.py                                        # Flask app entry point (1096 lines)
    ├── config.py                                     # Config classes (Dev/Prod/Test)
    ├── test_training.py                              # Quick ML training test
    │
    ├── database/
    │   ├── db_manager.py                             # SQLite database manager (388 lines)
    │   └── auth_system.db                            # SQLite database file
    │
    ├── models/
    │   ├── behavioral_models.py                      # All 6 ML models + ensemble (1009 lines)
    │   └── saved/                                    # Saved model files (per-user subdirs)
    │
    ├── utils/
    │   ├── feature_extractor.py                      # 38-feature extraction (830 lines)
    │   └── drift_detector.py                         # Statistical drift detection (475 lines)
    │
    ├── static/
    │   ├── css/styles.css                            # Dark theme UI (2803 lines)
    │   ├── js/login.js                               # Login/Register page logic (572 lines)
    │   ├── js/calib.js                               # Calibration wizard (808 lines)
    │   ├── js/challenge.js                           # Dashboard + duplicate code (1431 lines)
    │   └── plots/                                    # Generated evaluation plots (PNGs)
    │
    └── templates/
        ├── login.html                                # Login/Register page (258 lines)
        ├── calib.html                                # Calibration wizard (340 lines)
        └── challenge.html                            # Dashboard (407 lines)
```

## 3. Data Flow

### 3.1 Registration / Login (Initial Auth)

```
User enters credentials → login.js POST /api/login
  → app.py validates password via bcrypt
  → JWT access token created
  → SQLite session created
  → User redirected:
      If NOT calibrated → /calibration
      If calibrated     → /challenge
```

### 3.2 Calibration (Enrollment)

```
User enters /calibration → calib.js
  │
  ├── Step 1: Welcome (info screen)
  ├── Step 2: Typing Exercises (5 passages typed naturally)
  │     └── keystroke events captured → sent via Socket.IO
  ├── Step 3: Mouse Exercises (4 exercises):
  │     ├── Click Timing — click random appearing targets
  │     ├── Tracking — follow a moving circular target
  │     ├── Navigation — follow an animated SVG path
  │     └── Precision — click small static targets
  │     └── mouse events captured → sent via Socket.IO
  ├── Step 4: Complete
  │
  └── POST /api/calibration/complete
       → BehavioralData retrieved from DB
       → Features extracted via BehavioralFeatureExtractor
       → EnsembleBehavioralClassifier.train_initial_models() called
       → All 6 models trained on user's behavioral data
       → Drift detector baseline set
       → Models saved to disk (HDF5 + joblib pickles)
       → Calibration status updated in DB
       → Redirect to /challenge
```

### 3.3 Real-Time Monitoring (Dashboard)

```
User enters /challenge → challenge.js
  │
  ├── WebSocket connects → join_session emitted
  ├── Every keystroke/mouse event captured in buffer
  ├── Every 5 seconds:
  │     ├── Behavioral data sent via Socket.IO 'behavioral_data' event
  │     ├── app.py.handle_behavioral_data() runs:
  │     │     1. Features extracted via extractor
  │     │     2. Data stored in DB
  │     │     3. perform_real_time_authentication() called
  │     │         ├── Ensemble prediction (6 models)
  │     │         ├── Drift detection check
  │     │         ├── Alert level calculation
  │     │         ├── Incremental model update
  │     │         └── auth_result emitted back to client
  │     └── UI updated: auth score, confidence, anomaly risk, charts
  │
  ├── Cleanup session after 8h inactivity (background thread)
  └── Drift analysis available via 'request_drift_analysis' event
```

## 4. ML Models (Ensemble)

Six models, each contributing to a weighted ensemble score:

| Model | Weight | Type | Input | Output |
|-------|--------|------|-------|--------|
| **GRU** | 0.25 | Binary classification (sequence) | Sequences of 50 feature vectors | Authenticity score [0,1] |
| **Autoencoder** | 0.15 | Anomaly detection | Single feature vector | Reconstruction error → anomaly score |
| **One-Class SVM** | 0.15 | Outlier detection | Single feature vector | Outlier score [0,1] |
| **Incremental k-NN** | 0.20 | Sliding window k-NN | Single feature vector | Genuine vote ratio [0,1] |
| **Passive-Aggressive** | 0.15 | Online linear classifier | Single feature vector | Probability [0,1] |
| **Isolation Forest** | 0.10 | Anomaly detection | Single feature vector | Anomaly score [0,1] |

### 4.1 GRU Sequence Model (`GRUSequenceModel`)
- **Architecture**: Input(50, feature_dim) → GRU(64, return_sequences) → GRU(32) → Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, sigmoid)
- **Loss**: binary_crossentropy, **Metrics**: accuracy
- **Training**: Trains on genuine data (label=1) + optional imposter data (label=0)
- **Prediction**: Takes last sequence, outputs probability of being genuine

### 4.2 Autoencoder (`AutoencoderAnomalyDetector`)
- **Architecture**: Input(feature_dim) → Dense(16, ReLU) → Dropout(0.2) → Dense(8, ReLU) → Dense(16, ReLU) → Dropout(0.2) → Dense(feature_dim, sigmoid)
- **Loss**: MSE, **Metrics**: MAE
- **Threshold**: 95th percentile of training reconstruction errors
- **Anomaly Score**: `min(error / threshold, 1.0)`

### 4.3 One-Class SVM (`OneClassSVMDetector`)
- **Parameters**: nu=0.1, gamma='scale'
- **Training**: Fits on genuine data only
- **Prediction**: Decision function normalized to [0,1] outlier score

### 4.4 Incremental k-NN (`IncrementalKNNClassifier`)
- **Parameters**: k=5, window_size=1000
- **Training**: Stores genuine + imposter feature vectors in sliding window buffers
- **Prediction**: Euclidean distance to k-nearest neighbors, vote ratio

### 4.5 Passive-Aggressive (`PassiveAggressiveDetector`)
- **Parameters**: C=1.0
- **Training**: Online — `partial_fit()` with new samples
- **Prediction**: Decision function → sigmoid → probability

### 4.6 Isolation Forest (`IsolationForestDetector`)
- **Parameters**: contamination=0.1, n_estimators=100
- **Training**: Fits on genuine data
- **Prediction**: Decision function normalized to [0,1] anomaly score

### 4.7 Ensemble Score
Weighted average of all model predictions:
```
ensemble_score = Σ(weight_i * score_i * confidence_i) / Σ(weight_i * confidence_i)
consensus = 1 - min(std(scores), 1.0)
```

## 5. Feature Extraction (`BehavioralFeatureExtractor`)

Extracts exactly **38 features** — 18 keystroke + 20 mouse — from raw event data. Guarantees consistent dimensions via fixed feature definitions.

### 5.1 Keystroke Features (18)
| Feature | Description |
|---------|-------------|
| hold_time_mean / std / median | Key press duration statistics |
| flight_time_mean / std / median | Time between key release and next press |
| typing_speed_wpm / cpm | Words/characters per minute |
| rhythm_consistency | Inverse CV of inter-key intervals |
| burst_ratio / pause_ratio | Proportion of fast/slow intervals |
| avg_pause_duration | Mean pause length |
| speed_variance / speed_trend | Sliding window speed stats |
| digraph_consistency | Consistency of common letter pairs |
| hold_time_cv / flight_time_cv | Coefficient of variation |
| pressure_consistency | Inverse std of pressure (if available) |

### 5.2 Mouse Features (20)
| Feature | Description |
|---------|-------------|
| velocity_mean / std / median | Mouse speed statistics |
| acceleration_mean / std | Acceleration stats |
| movement_efficiency | Direct / total path distance ratio |
| curvature_mean / std | Path curvature at 3-point segments |
| avg_direction_change / direction_change_variance | Direction angle change stats |
| click_duration_mean / std | Click hold time stats |
| left_click_ratio / right_click_ratio | Click type proportions |
| inter_click_mean / std | Time between clicks |
| dwell_time_mean | Hover duration |
| movement_area / movement_centrality | Spatial spread of cursor |
| velocity_smoothness | Inverse mean jerk |

## 6. Database Schema (SQLite)

### 6.1 `users`
| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER PK | Auto-increment |
| username | TEXT UNIQUE | Login username |
| email | TEXT UNIQUE | User email |
| password_hash | TEXT | bcrypt hash |
| salt | TEXT | bcrypt salt |
| created_at | TIMESTAMP | Registration time |
| last_login | TIMESTAMP | Last login time |
| is_active | BOOLEAN | Account active flag |
| failed_attempts | INTEGER | Consecutive failed logins |
| locked_until | TIMESTAMP | Account lockout expiry |
| calibration_complete | BOOLEAN | Whether calibration done |

### 6.2 `sessions`
| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT PK | SHA-256 hash |
| user_id | INTEGER FK | References users |
| created_at | TIMESTAMP | Session start |
| last_activity | TIMESTAMP | Last activity time |
| is_active | BOOLEAN | Active flag |
| ip_address | TEXT | Client IP |
| user_agent | TEXT | Browser user agent |

### 6.3 `behavioral_data`
| Column | Type | Description |
|--------|------|-------------|
| data_id | INTEGER PK | Auto-increment |
| user_id | INTEGER FK | References users |
| session_id | TEXT FK | References sessions |
| timestamp | TIMESTAMP | Data capture time |
| data_type | TEXT | 'keystroke' or 'mouse' |
| features | TEXT (JSON) | Extracted feature dict (38 keys) |
| raw_data | TEXT (JSON) | Raw event data |
| confidence_score | REAL | ML confidence |
| anomaly_score | REAL | ML anomaly score |

### 6.4 `auth_events`
| Column | Type | Description |
|--------|------|-------------|
| event_id | INTEGER PK | Auto-increment |
| user_id | INTEGER FK | References users |
| session_id | TEXT FK | References sessions |
| event_type | TEXT | 'login', 'logout', 'anomaly', 'drift' |
| event_data | TEXT (JSON) | Details |
| timestamp | TIMESTAMP | Event time |
| ip_address | TEXT | Client IP |

### 6.5 `model_metadata`
| Column | Type | Description |
|--------|------|-------------|
| user_id | INTEGER PK FK | References users |
| model_version | INTEGER | Version counter |
| last_trained | TIMESTAMP | Last training time |
| training_samples | INTEGER | Number of training samples |
| model_accuracy | REAL | Last recorded accuracy |
| drift_detected | BOOLEAN | Drift status |
| drift_timestamp | TIMESTAMP | When drift was detected |

## 7. Drift Detection (`BehavioralDriftDetector`)

Detects if a user's behavioral patterns have changed significantly over time using:

- **Cohen's d** effect size for mean shifts
- **F-test ratio** for variance changes
- **Skewness/kurtosis** for distribution shape changes
- **KS-test**, **Mann-Whitney U**, **Levene's test** on demand
- Feature weighting (some features more important than others)

Drift score in [0,1]; thresholds: `>0.3` detected, `>0.5` severe → triggers retrain.

## 8. Configuration (`config.py` / `.env`)

### Key Parameters
| Parameter | Dev Default | Description |
|-----------|-------------|-------------|
| CONFIDENCE_THRESHOLD | 0.7 | Min confidence to trust auth |
| ANOMALY_SCORE_THRESHOLD | 0.8 | Threshold for anomaly alert |
| GRU_SEQUENCE_LENGTH | 50 | Sequences fed to GRU |
| DRIFT_ALPHA | 0.05 | Statistical significance level |
| DRIFT_DETECTION_WINDOW | 100 | Sliding window size |
| WINDOW_SIZE | 30s | Feature extraction window |
| JWT_ACCESS_TOKEN_EXPIRES | 24h | Token lifetime |
| SESSION_TIMEOUT | 8h | Session timeout |
| MAX_LOGIN_ATTEMPTS | 5 | Lockout threshold |

### Calibration Overrides (in app.py)
During calibration, these are made more lenient:
- `DRIFT_DETECTION_WINDOW = 20`
- `DRIFT_ALPHA = 0.1`
- `DRIFT_MIN_SAMPLES = 10`
- `DRIFT_THRESHOLD = 0.10`

## 9. Frontend Components

### 9.1 Login (`login.html` + `login.js`)
- LoginManager class handles login/register forms
- Password strength meter (6 criteria)
- Session persistence via localStorage
- Visibility change → re-check session
- POST `/api/login` and `/api/register`

### 9.2 Calibration (`calib.html` + `calib.js`)
- CalibrationManager class (4-step wizard)
- 5 typing passages + 4 mouse exercises
- Real-time stats (WPM, accuracy, samples, distance, clicks)
- WebSocket data transmission in batches (50 keystroke, 100 mouse)
- POST `/api/calibration/complete` on finish

### 9.3 Dashboard (`challenge.html` + `challenge.js`)
- DashboardManager class (1319 lines)
- 5 navigation sections: Dashboard, Security, Analytics, Activity, Settings
- 4 Chart.js charts (behavior line, drift radar, patterns scatter, time dual-axis)
- Real-time auth score, confidence, anomaly risk display
- Notification dropdown with security alerts
- Settings for auth threshold / anomaly sensitivity
- **Known issue**: Duplicate `DOMContentLoaded` block at lines 1324-1431

## 10. API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/register` | User registration |
| POST | `/api/login` | User login (returns JWT) |
| POST | `/api/logout` | End session |
| GET | `/api/session/status` | Check session validity |
| POST | `/api/calibration/complete` | Finalize calibration + train models |
| GET | `/api/debug/calibration-data/<id>` | (Debug) View calibration data |
| GET | `/api/debug/feature-dimensions/<id>` | (Debug) View feature info |

## 11. WebSocket Events (Socket.IO)

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client→Server | Initial connection |
| `join_session` | Client→Server | Authenticate + join room |
| `behavioral_data` | Client→Server | Send keystroke/mouse events |
| `request_drift_analysis` | Client→Server | Request drift status |
| `connected` | Server→Client | Connection confirmation |
| `session_joined` | Server→Client | Session auth success |
| `auth_result` | Server→Client | Real-time auth score |
| `security_alert` | Server→Client | Anomaly/threat alert |
| `drift_analysis` | Server→Client | Drift detection results |
| `session_error` | Server→Client | Session/auth error |
| `error` | Server→Client | Generic error |

## 12. Known Issues

1. **Duplicate code in challenge.js**: Two separate `DOMContentLoaded` listeners (lines 3-180, 1324-1431) both wire up event listeners and Socket.IO connections, causing double event handlers.
2. **Feature dimension mismatch potential**: `GRUSequenceModel.__init__` hardcodes `feature_dim=20` but `BehavioralFeatureExtractor` produces 38 features. The `_ensure_feature_consistency` method dynamically learns feature names, which mitigates this but may cause silent dimension mismatches.
3. **Duplicate DRIFT_ALPHA keys** in `config.py` (lines 43 and 64).
4. **Synthetic data fallback**: If calibration collects <20 samples, synthetic random data is generated, which may lead to poor model quality.
5. **GRU training requirement**: Requires >=50 feature samples to create at least one sequence (sequence_length=50). If fewer are available, GRU is skipped during training.
6. **LSP type errors**: Multiple non-blocking type-checking warnings throughout the codebase.

## 13. Dependencies (requirements.txt)
Flask, Flask-SocketIO, Flask-JWT-Extended, bcrypt, tensorflow, scikit-learn, numpy, scipy, joblib, python-dotenv, matplotlib, pandas

# Data Flow Analysis — Behavioral Authentication System

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            BROWSER (Frontend)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ login.js │  │ calib.js │  │ challenge.js  │  │   styles.css      │  │
│  │          │  │          │  │  (Dashboard)   │  │                   │  │
│  └────┬─────┘  └────┬─────┘  └──────┬────────┘  └───────────────────┘  │
│       │              │               │                                  │
│  HTTP POST       WebSocket       WebSocket + HTTP                     │
│       │              │               │                                  │
└───────┼──────────────┼───────────────┼──────────────────────────────────┘
        │              │               │
        ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FLASK APPLICATION (app.py)                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      ROUTE HANDLERS                               │   │
│  │  /api/register  /api/login  /api/logout  /api/calibration/complete│   │
│  │  /api/session/status  /login  /calibration  /challenge            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    WEBSOCKET HANDLERS (SocketIO)                   │   │
│  │  connect / disconnect / join_session / behavioral_data            │   │
│  │  request_drift_analysis / auth_result / security_alert            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────┬──────────────────┬──────────────────┐                │
│  │ In-Memory   │ Active Sessions  │ Behavioral       │                │
│  │ State:      │ Dict (session_id │ Buffers:         │                │
│  │             │ -> session_data) │ defaultdict of   │                │
│  │ user_models │                  │ {keystroke:      │                │
│  │ user_extra- │                  │  deque, mouse:   │                │
│  │ ctors       │                  │  deque, recent_  │                │
│  │ user_drift_ │                  │  features: deque}│                │
│  │ detectors   │                  │                  │                │
│  └─────────────┴──────────────────┴──────────────────┘                │
│                                                                         │
│  ┌──────────────────────┬──────────────────────┬─────────────────────┐ │
│  │  FeatureExtractor    │  EnsembleClassifier  │  DriftDetector      │ │
│  │  (utils/feature_     │  (models/behavioral_ │  (utils/drift_      │ │
│  │   extractor.py)      │   models.py)         │   detector.py)      │ │
│  │  38 features total:  │  - GRU               │  - Statistical      │ │
│  │  - 18 keystroke      │  - Autoencoder       │    tests (KS,       │ │
│  │  - 20 mouse          │  - OneClassSVM        │    Mann-Whitney,   │ │
│  │                      │  - IncrementalKNN     │    Levene)          │ │
│  │                      │  - PassiveAggressive  │  - Cohen's d        │ │
│  │                      │  - IsolationForest    │  - Feature weights  │ │
│  └──────────────────────┴──────────────────────┴─────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DATABASE MANAGER (db_manager.py)                │   │
│  │  Tables: users, sessions, behavioral_data, auth_events,           │   │
│  │          model_metadata                                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Data Flow by Phase

### 2.1 User Registration

```
[login.js]                          [app.py]                    [db_manager.py]
    │                                   │                             │
    │ POST /api/register               │                             │
    │ {username, email, password}       │                             │
    │──────────────────────────────────►│                             │
    │                                   │ create_user(username,       │
    │                                   │   email, password)          │
    │                                   │────────────────────────────►│
    │                                   │                             │
    │                                   │ 1. bcrypt.gensalt()          │
    │                                   │ 2. bcrypt.hashpw(pw, salt)  │
    │                                   │ 3. INSERT INTO users         │
    │                                   │ 4. INSERT INTO model_       │
    │                                   │    metadata (user_id, now)   │
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │ {success: true, user_id}          │                             │
    │◄──────────────────────────────────│                             │
    │                                   │                             │
    │ Save to localStorage: none        │                             │
    │ Show "Registration successful"    │                             │
    │ Switch back to login form         │                             │
```

### 2.2 Login

```
[login.js]                          [app.py]                    [db_manager.py]
    │                                   │                             │
    │ POST /api/login                   │                             │
    │ {username, password}               │                             │
    │──────────────────────────────────►│                             │
    │                                   │ authenticate_user(username, │
    │                                   │   password)                 │
    │                                   │────────────────────────────►│
    │                                   │                             │
    │                                   │ 1. SELECT user WHERE        │
    │                                   │    username=? AND is_active │
    │                                   │ 2. Check locked_until       │
    │                                   │ 3. bcrypt.checkpw(pw, hash) │
    │                                   │ 4. On fail: increment       │
    │                                   │    failed_attempts, may     │
    │                                   │    lock account (5 fails)   │
    │                                   │ 5. On success: reset fails, │
    │                                   │    update last_login        │
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │                                   │ create_session(...)          │
    │                                   │────────────────────────────►│
    │                                   │ 1. sha256(user_id+now+ip)   │
    │                                   │ 2. INSERT INTO sessions     │
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │                                   │ create_access_token(        │
    │                                   │   identity=user_id,         │
    │                                   │   claims={session_id})      │
    │                                   │                             │
    │                                   │ initialize_user_components  │
    │                                   │ (load ML models, extractors,│
    │                                   │  drift detectors for user)  │
    │                                   │                             │
    │                                   │ store in-memory:            │
    │                                   │ active_sessions[session_id] │
    │                                   │                             │
    │                                   │ log_auth_event('login')     │
    │                                   │────────────────────────────►│
    │                                   │                             │
    │ {success, access_token,           │                             │
    │  session_id, user_id, username,   │                             │
    │  redirect: /calibration or        │                             │
    │  /challenge}                      │                             │
    │◄──────────────────────────────────│                             │
    │                                   │                             │
    │ localStorage.setItem('access_token', ...)                       │
    │ localStorage.setItem('session_id', ...)                         │
    │ localStorage.setItem('user_id', ...)                            │
    │ localStorage.setItem('username', ...)                           │
    │                                                                 │
    │ window.location.href = data.redirect                            │
```

### 2.3 Calibration (Behavioral Data Collection)

**Phase A: Typing Exercise (Frontend → Backend via WebSocket)**

```
[calib.js]                          [app.py]                    [db_manager.py]
    │                                   │                             │
    │ === WebSocket Connection ===       │                             │
    │──────────────────────────────────►│                             │
    │ socket.emit('join_session',       │                             │
    │   {session_id})                   │                             │
    │──────────────────────────────────►│                             │
    │                                   │ authenticate_session(sid)    │
    │                                   │ join_room(session_id)        │
    │◄──────────────────────────────────│                             │
    │ 'session_joined'                  │                             │
    │                                   │                             │
    │ === User types passages ===       │                             │
    │                                   │                             │
    │ captureKeystroke(e):              │                             │
    │ {key, code, type, timestamp,      │                             │
    │  holdTime, ctrlKey, shiftKey}     │                             │
    │                                   │                             │
    │ Batch: when 50 events collected   │                             │
    │──────────────────────────────────►│                             │
    │ socket.emit('behavioral_data',    │                             │
    │   {type: 'keystroke',            │                             │
    │    events: [...],                 │                             │
    │    timestamp})                    │                             │
    │──────────────────────────────────►│                             │
    │                                   │                             │
    │                                   │ handle_behavioral_data():    │
    │                                   │ 1. authenticate_session()    │
    │                                   │ 2. initialize_user_         │
    │                                   │    components(user_id)       │
    │                                   │ 3. extractor.extract_       │
    │                                   │    keystroke_features(events)│
    │                                   │ 4. _normalize_keystroke_    │
    │                                   │    features(features) → 18   │
    │                                   │ 5. store in behavioral_buf  │
    │                                   │    fers[user_id]            │
    │                                   │ 6. store_behavioral_data(   │
    │                                   │    user_id, session_id,     │
    │                                   │    'keystroke', features,   │
    │                                   │    raw_events)              │
    │                                   │────────────────────────────►│
    │                                   │ INSERT INTO behavioral_data │
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │                                   │ (If session calibrated:     │
    │                                   │  perform_real_time_auth())  │
    │                                   │ (First time: no models yet) │
```

**Phase B: Mouse Exercise (Frontend → Backend via WebSocket)**

```
[calib.js]                          [app.py]                    [db_manager.py]
    │                                   │                             │
    │ captureMouseMovement(e):          │                             │
    │ {type, x, y, timestamp, velocity,│                             │
    │  distance, target}                │                             │
    │                                   │                             │
    │ captureMouseClick(e):             │                             │
    │ {type, button, x, y, timestamp,   │                             │
    │  eventType, duration, target}     │                             │
    │                                   │                             │
    │ Batch: when 100 events collected  │                             │
    │──────────────────────────────────►│                             │
    │ socket.emit('behavioral_data',    │                             │
    │   {type: 'mouse',                │                             │
    │    events: [...], timestamp})     │                             │
    │                                   │                             │
    │                                   │ handle_behavioral_data():    │
    │                                   │ 1. extract_mouse_features()  │
    │                                   │ 2. _normalize_mouse_         │
    │                                   │    features() → 20 features  │
    │                                   │ 3. buffer + DB storage       │
```

**Phase C: Calibration Completion (Frontend → Backend via HTTP)**

```
[calib.js]                          [app.py]                    [db_manager.py]
    │                                   │                             │
    │ POST /api/calibration/complete    │                             │
    │ {session_id}                      │                             │
    │──────────────────────────────────►│                             │
    │                                   │                             │
    │                                   │ 1. authenticate_session()    │
    │                                   │                             │
    │                                   │ 2. get_user_behavioral_data │
    │                                   │    (user_id, 'keystroke',    │
    │                                   │     limit=1000)              │
    │                                   │────────────────────────────►│
    │                                   │ SELECT * FROM behavioral_   │
    │                                   │   data WHERE user_id=? AND  │
    │                                   │   data_type='keystroke' ... │
    │                                   │◄────────────────────────────│
    │                                   │ (same for 'mouse')          │
    │                                   │                             │
    │                                   │ 3. If < 20 total samples:   │
    │                                   │    generate_synthetic_      │
    │                                   │    behavioral_data(user_id)  │
    │                                   │    → 30 fake keystroke +    │
    │                                   │      30 fake mouse entries  │
    │                                   │                             │
    │                                   │ 4. initialize_user_          │
    │                                   │    components(user_id)       │
    │                                   │                             │
    │                                   │ 5. For each stored item:    │
    │                                   │    extract features from    │
    │                                   │    'features' JSON field or │
    │                                   │    raw_data                 │
    │                                   │                             │
    │                                   │ 6. If <10 features in any   │
    │                                   │    category: create_minimal_│
    │                                   │    keystroke/mouse_features │
    │                                   │    → 15 synthetic samples   │
    │                                   │                             │
    │                                   │ 7. Combine all features     │
    │                                   │    → train initial models   │
    │                                   │                             │
    │                                   │ 8. drift_detector.set_      │
    │                                   │    reference_baseline(       │
    │                                   │    keystroke_features,       │
    │                                   │    mouse_features)           │
    │                                   │                             │
    │                                   │ 9. user_models.save_all_    │
    │                                   │    models() → HDF5/pkl      │
    │                                   │    files at                 │
    │                                   │    models/saved/{user_id}/  │
    │                                   │                             │
    │                                   │ 10. update_calibration_     │
    │                                   │     status(user_id, True)   │
    │                                   │────────────────────────────►│
    │                                   │ UPDATE users SET            │
    │                                   │   calibration_complete=1    │
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │                                   │ update_model_metadata(...)   │
    │                                   │────────────────────────────►│
    │                                   │ UPDATE model_metadata SET   │
    │                                   │   accuracy, training_samples│
    │                                   │◄────────────────────────────│
    │                                   │                             │
    │ {success, training_results:       │                             │
    │  {accuracy, keystroke_samples,    │                             │
    │   mouse_samples, total_samples,   │                             │
    │   models_trained},                │                             │
    │  redirect: '/challenge'}          │                             │
    │◄──────────────────────────────────│                             │
```

---

## 3. Real-Time Authentication Data Flow (Post-Calibration)

This is the core continuous authentication loop that runs during active sessions:

```
[BROWSER challenge.js]          [app.py]                  [models/behavioral_models.py]
         │                           │                               │
         │ === Every keystroke/mouse event ===                       │
         │                           │                               │
         │ Batched WebSocket emit:   │                               │
         │ behavioral_data event     │                               │
         │──────────────────────────►│                               │
         │                           │                               │
         │                           │ handle_behavioral_data():     │
         │                           │  1. authenticate_session()    │
         │                           │  2. initialize_user_          │
         │                           │     components(user_id)       │
         │                           │  3. Extract features via      │
         │                           │     feature_extractor:         │
         │                           │     - extract_keystroke_      │
         │                           │       features(raw_events)     │
         │                           │       → 18-feature dict       │
         │                           │     - _normalize_keystroke_   │
         │                           │       features()              │
         │                           │     - OR extract_mouse_       │
         │                           │       features() → 20-feature │
         │                           │       dict + normalize        │
         │                           │                               │
         │                           │  4. buffer.append(features)   │
         │                           │     buffer.append(raw_events) │
         │                           │                               │
         │                           │  5. store_behavioral_data(    │
         │                           │     features → DB)            │
         │                           │──────────► db_manager         │
         │                           │                               │
         │                           │  6. If calibrated:            │
         │                           │     perform_real_time_        │
         │                           │     authentication(user_id,   │
         │                           │     features, data_type)      │
         │                           │──────────────────────────────►│
         │                           │                               │
         │                           │   a. Get recent_features      │
         │                           │      from buffer (max 100)    │
         │                           │                               │
         │                           │   b. For each feature dict:   │
         │                           │      fix_feature_dimensions() │
         │                           │      → ensure exactly 38      │
         │                           │        features total         │
         │                           │                               │
         │                           │   c. ensemble.predict_        │
         │                           │      ensemble(features)       │
         │                           │──────────────────────────────►│
         │                           │                               │
         │                           │    GRU: predict(features)      │
         │                           │    1. _ensure_feature_         │
         │                           │       consistency(features)   │
         │                           │    2. prepare_sequences():    │
         │                           │       convert → matrix →      │
         │                           │       scaler.transform →      │
         │                           │       sliding windows of      │
         │                           │       50 samples each         │
         │                           │    3. model.predict(seq)      │
         │                           │    4. Return (score[0-1],     │
         │                           │       confidence[0-1])        │
         │                           │                               │
         │                           │    Autoencoder: predict_      │
         │                           │    anomaly_score(features)     │
         │                           │    1. prepare_data():         │
         │                           │       ensure consistency →    │
         │                           │       matrix → scaler         │
         │                           │    2. model.reconstruct(X)    │
         │                           │    3. MSE reconstruction err  │
         │                           │    4. Normalize by threshold  │
         │                           │    5. Return anomaly[0-1]    │
         │                           │                               │
         │                           │    OneClassSVM: predict_      │
         │                           │    outlier_score(features)     │
         │                           │    1. prepare_data()          │
         │                           │    2. decision_function(X)    │
         │                           │    3. Normalize [-2,2] → [0,1]│
         │                           │    4. Return outlier_score    │
         │                           │                               │
         │                           │    k-NN: predict(features)    │
         │                           │    1. Euclidean distance to   │
         │                           │       all stored samples      │
         │                           │    2. k=5 nearest neighbors   │
         │                           │    3. Vote: genuine vs impost │
         │                           │    4. Return (score, conf)    │
         │                           │                               │
         │                           │    PassiveAggressive:         │
         │                           │    predict(features)           │
         │                           │    1. scaler.transform(X)     │
         │                           │    2. decision_function(X)    │
         │                           │    3. Sigmoid → probability   │
         │                           │    4. Return (probability,    │
         │                           │       confidence)             │
         │                           │                               │
         │                           │    IsolationForest: predict_  │
         │                           │    anomaly_score(features)     │
         │                           │    1. prepare_data()          │
         │                           │    2. decision_function(X)    │
         │                           │    3. Normalize [-1,1] → [0,1]│
         │                           │    4. Return anomaly_score    │
         │                           │                               │
         │                           │◄──────────────────────────────│
         │                           │                               │
         │                           │   d. _calculate_ensemble_     │
         │                           │      score():                 │
         │                           │     - GRU: weight 0.25 * conf │
         │                           │     - Autoencoder: weight 0.15│
         │                           │       (1 - anomaly → auth)    │
         │                           │     - SVM: weight 0.15        │
         │                           │       (1 - outlier → auth)    │
         │                           │     - k-NN: weight 0.20 * conf│
         │                           │     - PA: weight 0.15 * conf  │
         │                           │     - IF: weight 0.10         │
         │                           │       (1 - anomaly → auth)    │
         │                           │     → weighted avg auth_score │
         │                           │                               │
         │                           │   e. anomaly_score =          │
         │                           │      1.0 - auth_score         │
         │                           │                               │
         │                           │   f. drift_detector.add_      │
         │                           │      sample(features, type)    │
         │                           │     → _check_for_drift():      │
         │                           │       Cohen's d mean shift    │
         │                           │       F-test variance change  │
         │                           │       Skew/kurtosis change    │
         │                           │       → drift_score [0-1]     │
         │                           │                               │
         │                           │   g. Determine alert level:   │
         │                           │     score > threshold &       │
         │                           │     conf > 0.7 → level 3     │
         │                           │     conf > 0.5 → level 2     │
         │                           │     else → level 1           │
         │                           │     + drift detected → level 1│
         │                           │                               │
         │                           │   h. update_models(features,  │
         │                           │      is_genuine=True)         │
         │                           │     → k-NN buffer update      │
         │                           │     → PA partial_fit          │
         │                           │                               │
         │ ◄─────────────────────────│                               │
         │ emit('auth_result', {     │                               │
         │   authenticity_score,     │                               │
         │   confidence, consensus,  │                               │
         │   anomaly_detected,       │                               │
         │   anomaly_score,          │                               │
         │   alert_level,            │                               │
         │   alert_message,          │                               │
         │   recommendations,        │                               │
         │   drift_analysis})         │                               │
         │                           │                               │
         │ handleAuthResult(data):   │                               │
         │ Update UI dashboard:      │                               │
         │ - authScore display       │                               │
         │ - confidence level %      │                               │
         │ - anomaly risk (Low/Med/  │                               │
         │   High)                   │                               │
         │ - status dot color        │                               │
         │   (green/yellow/red)      │                               │
         │ - behavior chart update   │                               │
         │ - time chart update        │                               │
```

---

## 4. Feature Extraction Pipeline (Detailed)

### 4.1 Keystroke Features (18)

```
Raw keystroke events:
[{key: 'a', type: 'keydown', timestamp: T1, hold_time: ...},
 {key: 'a', type: 'keyup', timestamp: T2, flight_time: ...}, ...]

                    │
                    ▼
BehavioralFeatureExtractor.extract_keystroke_features(events)

    _extract_timing_stats('hold_time', hold_times)
    ├── hold_time_mean:   μ of key-down-to-key-up durations
    ├── hold_time_std:    σ of hold times
    └── hold_time_median: P50 of hold times

    _extract_timing_stats('flight_time', flight_times)
    ├── flight_time_mean:   μ of key-up-to-next-key-down intervals
    ├── flight_time_std:    σ of flight times
    └── flight_time_median: P50 of flight times

    _extract_speed_features(events, timestamps)
    ├── typing_speed_wpm: chars/sec / 5 * 60
    ├── typing_speed_cpm: chars/sec * 60
    ├── speed_variance:   var of sliding window WPM
    └── speed_trend:      linear regression slope

    _extract_rhythm_features(events, timestamps)
    ├── rhythm_consistency:   1 / (1 + CV of inter-key intervals)
    ├── burst_ratio:          fraction of intervals < P25
    ├── pause_ratio:          fraction of intervals > P75
    └── avg_pause_duration:   μ of pause-length intervals

    _extract_consistency_features(events)
    ├── digraph_consistency:  1 / (1 + σ of per-digraph timings)
    ├── hold_time_cv:         σ / μ of hold times
    └── flight_time_cv:       σ / μ of flight times

    _extract_pressure_features(events)
    └── pressure_consistency: 1 / (1 + σ of pressure values)
                               (default 0.8 if no pressure data)

    _normalize_keystroke_features(features)
    └── Enforces exactly 18 keys with defaults for missing values
```

### 4.2 Mouse Features (20)

```
Raw mouse events:
[{type: 'move', x: X1, y: Y1, timestamp: T1, velocity: V1},
 {type: 'click', button: 0, x: X2, y: Y2, duration: D1}, ...]

                    │
                    ▼
BehavioralFeatureExtractor.extract_mouse_features(events)

    _extract_movement_features(events)
    ├── velocity_mean:       μ of velocities (computed from dx/dt if not provided)
    ├── velocity_std:        σ of velocities
    ├── velocity_median:     P50 of velocities
    ├── acceleration_mean:   μ of dv/dt
    ├── acceleration_std:    σ of accelerations
    ├── movement_efficiency: straight-line dist / total path dist
    └── velocity_smoothness: 1 / (1 + mean|jerk|) where jerk = d²v/dt²

    _extract_click_features(events)
    ├── click_duration_mean:  μ of click hold durations
    ├── click_duration_std:   σ of click durations
    ├── left_click_ratio:     button==0 / total clicks
    ├── right_click_ratio:    button==2 / total clicks
    ├── inter_click_mean:     μ of time between clicks
    └── inter_click_std:      σ of inter-click intervals

    _extract_trajectory_features(events)
    ├── curvature_mean:              μ of curvature at each point
    ├── curvature_std:               σ of curvatures
    ├── avg_direction_change:        μ of angular changes between segments
    └── direction_change_variance:   var of angular changes

    _extract_behavioral_patterns(events)
    ├── dwell_time_mean:       μ of hover durations (type=='hover')
    ├── movement_area:         (maxX-minX)*(maxY-minY)
    └── movement_centrality:   μ of distances from centroid

    _normalize_mouse_features(features)
    └── Enforces exactly 20 keys with defaults for missing values
```

### 4.3 Combined Feature Vector (38)

```
get_combined_features(keystroke_dict, mouse_dict):
    1. _normalize_keystroke_features(ks)  → 18 guaranteed keys
    2. _normalize_mouse_features(mouse)    → 20 guaranteed keys
    3. combined = {**ks_normalized, **mouse_normalized}  → 38 keys
    4. assert len(combined) == 38

get_feature_vector(ks_dict, mouse_dict):
    combined = get_combined_features(...)
    return np.array([combined[name] for name in ALL_FEATURES], dtype=f32)
```

---

## 5. ML Model Training Pipeline (Calibration Completion)

```
Training flow within EnsembleBehavioralClassifier.train_initial_models(features):

                  All features (list of 38-feature dicts)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
    GRU Model         Autoencoder          OneClassSVM
    ─────────         ──────────           ───────────
    Requires ≥50      Requires ≥20         Requires ≥10
    samples           samples              samples
        │                   │                   │
    1. _ensure_feature_  1. _ensure_feature_ 1. _ensure_feature_
       consistency()       consistency()       consistency()
    2. prepare_sequences 2. prepare_data()   2. prepare_data()
       → matrix               → matrix            → matrix
       → scaler.fit()         → scaler.fit()      → scaler.fit()
       → sliding windows      → scaler.transform()  → scaler.transform()
         of 50 samples
                          3. model.fit(X, X)   3. model.fit(X)
    3. model.fit(X, y)       → reconstruct         → one-class boundary
       → binary (1=genuine)    → MSE error
                           4. threshold = P95   4. inlier_ratio calc
    4. metrics: accuracy,      of train errors
       precision, recall,                          → {'inlier_ratio',
       loss               → {'loss', 'threshold',    'support_vectors'}
                             'mean_reconstruction
                              _error'}

        ▼                   ▼                   ▼
    k-NN Classifier     IsolationForest     PassiveAggressive
    ──────────────      ───────────────     ─────────────────
    No minimum          Requires ≥20        Requires ≥5
    (incremental)       samples             samples
        │                   │                   │
    For each feature:   1. prepare_data()    1. prepare_data()
    update(feat,           → matrix              → matrix
      is_genuine=True)      → scaler.fit()       → scaler.fit()
                            → scaler.transform() → scaler.transform()
                         2. model.fit(X)      2. partial_fit(X, y)
                         3. inlier_ratio        → online learning
                             calculation          → classes=[0,1]
                          → {'inlier_ratio'}    → {'initialized': True}
```

---

## 6. Drift Detection Data Flow

```
DriftDetector maintains:
  - keystroke_window: deque(maxlen=100) of recent keystroke feature dicts
  - mouse_window: deque(maxlen=100) of recent mouse feature dicts
  - reference_keystroke: stats dict from calibration
  - reference_mouse: stats dict from calibration

Every add_sample(features, data_type):
    1. Append features to appropriate sliding window
    2. If window has ≥30 samples → _check_for_drift()

_check_for_drift():
    For each data type with reference + sufficient samples:
        1. current_stats = _calculate_feature_statistics(window)
           → {feature: {mean, std, median, q25, q75, skewness, kurtosis}}
        2. For each feature in reference:
             _calculate_feature_drift(current, reference, name):
                 mean_drift = Cohen's d effect size (weight 0.4)
                 variance_drift = |log(var_ratio)| / log(2) (weight 0.3)
                 shape_drift = avg(skew_diff/2, kurt_diff/2) (weight 0.3)
                 → drift_score [0, 1]
             Apply feature weight from feature_weights dict
        3. overall_drift = mean of all weighted feature drifts
        4. If overall_drift > 0.3 → set drift_detected = True

Formal statistical tests (available separately):
  - KS test: distribution comparison
  - Mann-Whitney U: median comparison
  - Levene's test: variance comparison
```

---

## 7. Database Schema and Data Persistence

### Tables

| Table | Key Fields | Purpose |
|-------|-----------|---------|
| `users` | user_id, username, email, password_hash, salt, calibration_complete, failed_attempts, locked_until | User accounts |
| `sessions` | session_id, user_id, created_at, last_activity, is_active, ip_address, user_agent | Active session tracking |
| `behavioral_data` | data_id, user_id, session_id, timestamp, data_type, features (JSON), raw_data (JSON), confidence_score, anomaly_score | Behavioral feature storage |
| `auth_events` | event_id, user_id, session_id, event_type, event_data (JSON), timestamp, ip_address | Audit log |
| `model_metadata` | user_id, model_version, last_trained, training_samples, model_accuracy, drift_detected | Model state tracking |

### Data Flow to Disk

```
                           behavioral_data
                         ┌─────────────────┐
  WebSocket event ──────►│ features: JSON  │
                         │ raw_data: JSON  │
                         │ data_type: str  │
                         └─────────────────┘
                                    │
                           auth_events
                         ┌─────────────────┐
  anomaly/login/logout──►│ event_type: str │
                         │ event_data: JSON │
                         └─────────────────┘
                                    │
                           model_metadata
                         ┌─────────────────┐
  calibration complete──►│ accuracy: float │
                         │ samples: int    │
                         └─────────────────┘
                                    │
                      HDF5 / pickle files
                    models/saved/{user_id}/
                    ├── model_gru.h5
                    ├── model_gru_scaler.pkl
                    ├── model_autoencoder.h5
                    ├── model_autoencoder_params.pkl
                    ├── model_svm.pkl
                    ├── model_knn.pkl
                    ├── model_pa.pkl
                    └── model_isolation.pkl
```

---

## 8. In-Memory State (app.py globals)

| Data Structure | Key | Value | Purpose |
|---------------|-----|-------|---------|
| `active_sessions` | session_id (str) | {user_id, username, session_id, login_time, last_activity, calibration_complete} | Track active sessions for fast lookup |
| `user_models` | user_id (int) | EnsembleBehavioralClassifier instance | Cached ML models per user |
| `user_extractors` | user_id (int) | BehavioralFeatureExtractor instance | Cached feature extractors |
| `user_drift_detectors` | user_id (int) | BehavioralDriftDetector instance | Cached drift detectors |
| `behavioral_buffers` | user_id (int) | {keystroke: deque(1000), mouse: deque(1000), recent_features: deque(100)} | Real-time data accumulation |

---

## 9. End-to-End Flow Summary

```
User Action              Frontend                Backend                   Database
══════════════           ════════════            ═══════════               ══════════
Register                 POST /api/register      create_user()             INSERT users
                                                                           INSERT model_metadata
Login                    POST /api/login         authenticate_user()       SELECT/UPDATE users
                                                 create_session()          INSERT sessions
                                                 create_access_token()
                                                 load ML models
                          Save JWT to localStorage
                          Redirect to /calibration or /challenge

Navigate to              GET /calibration        render_template(calib.html)
Calibration               Show typing passages

Type passages            socket.emit(            handle_behavioral_data()  INSERT behavioral_data
                          'behavioral_data',      extract_keystroke_
                          keystroke events)        features()
                                                  buffer events in memory

Mouse exercises          socket.emit(            handle_behavioral_data()  INSERT behavioral_data
                          'behavioral_data',      extract_mouse_features()
                          mouse events)           buffer events

Complete Calibration     POST /api/calibration/  Fetch stored behavioral   SELECT behavioral_data
                          complete                data from DB             UPDATE users (calibrated)
                         (session_id)             If <20 samples:          UPDATE model_metadata
                                                  generate synthetic data   SAVE HDF5 files
                                                  train all 6 ML models
                                                  set drift baseline
                                                  update calibration status

Navigate to              GET /challenge          render_template(challenge.html)
Dashboard (challenge)     Start real-time monitoring

Ongoing behavior         socket.emit(            handle_behavioral_data()  INSERT behavioral_data
(key + mouse)             'behavioral_data',      extract features
                          batched events)         buffer features

Every 5s                                          perform_real_time_
                                                  authentication():
                                                  1. Gather recent 38-dim
                                                     feature vectors
                                                  2. Ensemble predict:
                                                     GRU → auth_score
                                                     AE → anomaly_score
                                                     SVM → outlier_score
                                                     k-NN → auth_score
                                                     PA → auth_score
                                                     IF → anomaly_score
                                                  3. Weighted ensemble score
                                                  4. drift_detector.add_
                                                     sample()
                                                  5. Calculate alert level

                          socket.on(             emit('auth_result')       
                          'auth_result')           to client
                          Update dashboard UI

Logout                   POST /api/logout        end_session()             UPDATE sessions
                                                 log_auth_event()          INSERT auth_events
                          Clear localStorage
                          Redirect to /login
```

---

## 10. Key Design Decisions

1. **Hybrid communication**: HTTP for request-response (login, register, calibration complete), WebSocket for streaming (behavioral data, real-time auth results).

2. **Feature dimension guarantees**: The `BehavioralFeatureExtractor` enforces exactly 18 keystroke + 20 mouse = 38 total features via `_normalize_*_features()` methods, filling missing features with domain-appropriate defaults.

3. **Synthetic data fallback**: During calibration, if the user provides fewer than 20 behavioral samples, the system generates synthetic data with realistic random ranges to bootstrap training.

4. **Incremental learning**: `IncrementalKNNClassifier` and `PassiveAggressiveDetector` support online updates via `update_models()` called after every real-time authentication, allowing the models to adapt to gradual behavioral changes.

5. **Sliding windows everywhere**: Both frontend (30s sliding window for authentication data send) and backend (deques with maxlen=100 for recent features, 1000 for raw events) use sliding windows to bound memory usage.

6. **Graceful degradation**: Every ML prediction in the ensemble is wrapped in try/except blocks. If any model fails, it defaults to a score of 0.5 (neutral). If the entire ensemble fails, a random score around 0.7 is used as fallback.

# Locksy — Continuous Behavioral Authentication

A  standalone desktop application that continuously authenticates the user by analyzing keystroke dynamics and mouse behavior using an ensemble of machine learning models. All data stays on the local machine — no network dependencies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Feature Extraction](#feature-extraction)
- [Security Features](#security-features)
- [Database Schema](#database-schema)
- [Packaging](#packaging)

## Overview

Locksy replaces traditional password-based authentication with continuous behavioral biometrics. It monitors how you type and move your mouse, builds a unique behavioral profile using ML, and locks your workstation if anomalous behavior is detected. Everything runs locally — no data ever leaves your machine.

## Features

- **Behavioral Biometric Analysis**: Captures keystroke timing and mouse movement patterns via OS-level hooks (pynput)
- **Continuous Authentication**: Real-time verification every 5 seconds during active use
- **Multi-Model Ensemble**: GRU neural network, autoencoder, One-Class SVM, incremental k-NN, Passive-Aggressive classifier, and Isolation Forest
- **OS-Level Lock**: Workstation lockdown on sustained anomalies — Windows (LockWorkStation), macOS (CGSession), Linux (xdg-screensaver)
- **Model Integrity Verification**: SHA-256 hashing with HMAC signing to detect model file tampering
- **Encrypted Local Storage**: SQLCipher for the SQLite database (falls back to plain SQLite if unavailable)
- **Offline-First**: Zero networking — no Flask, no WebSockets, no JWT, no HTTP
- **Single-User**: Simplified single-profile design, no registration or login

## Technical Stack

### Desktop Framework
- **Eel**: Python-to-JS bridge for the desktop UI (Chrome/Edge embedded)
- **pynput**: OS-level global keyboard and mouse capture

### ML/AI
- **TensorFlow/Keras**: GRU networks, autoencoders
- **scikit-learn**: One-Class SVM, Isolation Forest, k-NN, Passive-Aggressive
- **NumPy, SciPy, Pandas**: Numerical computing and statistics

### Storage & Integrity
- **SQLite / pysqlcipher3**: Local encrypted database
- **joblib**: Model serialization
- **HMAC-SHA256**: Model file integrity verification

### Packaging
- **PyInstaller**: Bundles everything into a standalone executable

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Desktop Window (Eel)                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │ calib.html   │    │  challenge.html  │    │ Chart.js       │  │
│  │ calib.js     │    │  challenge.js    │    │ Dashboard      │  │
│  └──────┬───────┘    └────────┬─────────┘    └────────────────┘  │
│         │                     │                                   │
│         └─────────┬───────────┘                                   │
│                   │  eel.expose() / eel.callback()                │
└───────────────────┼───────────────────────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────────────────────┐
│                         app.py (Eel entry point)                   │
│                                                                   │
│  ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐  │
│  │ pynput       │    │  Monitoring Loop   │    │ Eel Exposed   │  │
│  │ Keyboard/    │───►│  (every 5s):       │    │ Functions:    │  │
│  │ Mouse        │    │  buffer → extract  │    │ • calibration │  │
│  │ Listeners    │    │  → ensemble → emit │    │ • auth check  │  │
│  └──────┬───────┘    │  → lock if needed  │    │ • activity log│  │
│         │            └────────┬───────────┘    └───────────────┘  │
│         ▼                     │                                    │
│  ┌──────────────┐            │                                    │
│  │ Feature      │◄───────────┘                                    │
│  │ Buffer       │                                                  │
│  │ (deques)     │                                                  │
│  └──────────────┘                                                  │
└───────────────────┬───────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────────────┐
    │               │                       │
┌───▼──────────┐ ┌──▼──────────────┐  ┌────▼──────────────┐
│ Feature      │ │ ML Ensemble     │  │ Drift Detector    │
│ Extractor    │ │ (6 models)      │  │ (statistical)     │
│ (38 features)│ │                 │  │                   │
└──────────────┘ └─────────────────┘  └───────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Database Manager (SQLite/SQLCipher) │
│  ┌──────────┐ ┌──────────┐          │
│  │ profile  │ │behavioral│          │
│  │          │ │_data     │          │
│  ├──────────┤ ├──────────┤          │
│  │auth_event│ │model_    │          │
│  │s         │ │metadata  │          │
│  └──────────┘ └──────────┘          │
└──────────────────────────────────────┘
```

## Project Structure

```
Behavior_based_Auth/
├── app.py                          # Eel desktop entry point
├── config.py                       # DesktopConfig (paths, thresholds)
├── requirements.txt                # Dependencies
├── locksy.spec                     # PyInstaller spec file
│
├── database/
│   └── db_manager.py              # SQLite/SQLCipher database (single-user)
│
├── models/
│   ├── behavioral_models.py       # 6 ML models + EnsembleBehavioralClassifier
│   └── saved/                     # Trained model files
│
├── utils/
│   ├── feature_extractor.py       # 38-feature extraction (18 keystroke + 20 mouse)
│   ├── drift_detector.py          # Statistical behavioral drift detection
│   ├── locks.py                   # OS-level workstation lock
│   └── security.py                # SHA-256 model hashing and integrity verification
│
└── web/
    ├── calib.html                 # Calibration wizard UI
    ├── challenge.html             # Dashboard / monitoring UI
    ├── css/
    │   └── styles.css             # Application styling
    └── js/
        ├── calib.js               # Calibration data collection
        └── challenge.js           # Dashboard, charts, real-time updates
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Chrome or Edge (for Eel's browser window)

### Steps

1. **Clone and enter the project**
```bash
cd Behavior_based_Auth
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

The app will check if calibration has been completed. If not, it shows the calibration wizard. After calibration, it shows the live monitoring dashboard.

## Configuration

Edit `config.py` (`DesktopConfig` class) to customize behavior:

```python
# Paths
DATABASE_PATH         # Location of the SQLite database
MODELS_BASE_PATH      # Where trained models are stored

# Authentication Thresholds
ANOMALY_SCORE_THRESHOLD   = 0.8   # Anomaly score triggers alert
CONFIDENCE_THRESHOLD      = 0.7   # Minimum confidence for acceptance
CONSECUTIVE_ANOMALIES_LIMIT = 3   # Lockdown trigger count

# ML Model Parameters
GRU_SEQUENCE_LENGTH     = 50     # Sequences fed to GRU
GRU_HIDDEN_UNITS        = 64     # GRU layer size
AUTOENCODER_ENCODING_DIM = 32    # Autoencoder bottleneck

# Drift Detection
DRIFT_DETECTION_WINDOW  = 100    # Sliding window size
DRIFT_ALPHA             = 0.05   # Statistical significance
DRIFT_MIN_SAMPLES       = 30     # Minimum samples for drift check

# Buffer Sizes
KEYSTROKE_BUFFER_SIZE   = 1000   # Max keystroke events in memory
MOUSE_BUFFER_SIZE       = 2000   # Max mouse events in memory
FEATURE_UPDATE_INTERVAL = 5      # Monitoring loop interval (seconds)
```

### Environment Variables
- `LOCKSY_DB_KEY`: Encryption key for the SQLCipher database (if pysqlcipher3 is installed)

## Usage

### First Run — Calibration

1. Launch the app: `python app.py`
2. The calibration wizard opens automatically
3. **Typing exercises**: Type 5 passages naturally (WPM, accuracy, samples shown)
4. **Mouse exercises**: Complete 4 exercises — target clicking, tracking, navigation, precision
5. Click **Complete Calibration** — the ensemble of 6 models trains on your data
6. Models are saved to disk with integrity hashes
7. You're redirected to the monitoring dashboard

### Ongoing Monitoring — Dashboard

The dashboard shows:
- **Security score** and **authentication score** (0-1)
- **Confidence level** and **anomaly risk** (Low/Medium/High)
- **Real-time behavior chart** (auth score over time)
- **Drift analysis** (radar chart comparing current vs. baseline)
- **Activity log** with filterable event history
- **Settings** for authentication threshold and anomaly sensitivity

### What Happens During an Anomaly

1. pynput captures every keystroke and mouse event
2. Every 5 seconds, the monitoring loop extracts 38 features from the event buffer
3. All 6 ensemble models predict authenticity
4. Weighted ensemble score is computed
5. If anomaly score exceeds threshold:
   - **1st occurrence**: Level 1 alert (low confidence)
   - **2nd occurrence**: Level 2 alert (high confidence)
   - **3rd consecutive occurrence**: **Workstation lockdown** via OS API

## Machine Learning Models

| Model | Weight | Type | Description |
|-------|--------|------|-------------|
| **GRU** | 0.25 | Sequential RNN | Binary classifier on 50-sample sequences of 38-dim features |
| **Autoencoder** | 0.15 | Anomaly detection | Reconstruction error threshold (95th percentile) |
| **One-Class SVM** | 0.15 | Outlier detection | Decision boundary around genuine data |
| **Incremental k-NN** | 0.20 | Sliding window | k=5 nearest neighbors vote ratio |
| **Passive-Aggressive** | 0.15 | Online linear | Partial-fit classifier for incremental learning |
| **Isolation Forest** | 0.10 | Anomaly detection | Isolation path length anomaly scoring |

### Ensemble Score

```
ensemble_score = Σ(weight_i × score_i × confidence_i) / Σ(weight_i × confidence_i)
consensus = 1 - min(std(scores), 1.0)
```

## Feature Extraction

`BehavioralFeatureExtractor` produces exactly **38 features** (18 keystroke + 20 mouse):

### Keystroke Features (18)
| Category | Features |
|----------|----------|
| Hold time | mean, std, median |
| Flight time | mean, std, median |
| Speed | WPM, CPM |
| Rhythm | consistency, burst ratio, pause ratio, avg pause duration |
| Variation | speed variance, speed trend |
| Consistency | digraph consistency, hold time CV, flight time CV |
| Pressure | pressure consistency |

### Mouse Features (20)
| Category | Features |
|----------|----------|
| Velocity | mean, std, median |
| Acceleration | mean, std |
| Trajectory | movement efficiency, curvature mean/std |
| Direction | avg direction change, direction change variance |
| Clicks | click duration mean/std, left/right click ratio |
| Inter-click | inter-click mean/std |
| Dwell | dwell time mean |
| Spatial | movement area, movement centrality |
| Smoothness | velocity smoothness |

## Security Features

- **No network stack**: Zero listening ports, no HTTP/WebSocket servers, no external API calls
- **Model integrity verification**: HMAC-SHA256 hashes of all model files; verified on startup; app refuses to start monitoring if check fails
- **Encrypted local storage**: SQLCipher database encryption (optional — falls back to plain SQLite if pysqlcipher3 unavailable)
- **OS-level lock**: Windows `LockWorkStation`, macOS `CGSession -suspend`, Linux `xdg-screensaver lock`
- **Event logging**: All anomaly detections and security events logged to database and file
- **No raw content capture**: pynput listeners capture only timing metadata (key press/release timestamps), never the content of what was typed

## Database Schema

### `profile`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Always 1 (single-user) |
| username | TEXT | Display name |
| calibration_complete | INTEGER | 0 or 1 |
| created_at | TIMESTAMP | |

### `behavioral_data`
| Column | Type | Description |
|--------|------|-------------|
| data_id | INTEGER PK | |
| timestamp | TIMESTAMP | |
| data_type | TEXT | 'keystroke' or 'mouse' |
| features | TEXT (JSON) | Extracted 38-feature dict |
| raw_data | TEXT (JSON) | Raw event data |

### `auth_events`
| Column | Type | Description |
|--------|------|-------------|
| event_id | INTEGER PK | |
| event_type | TEXT | 'anomaly', 'drift', etc. |
| event_data | TEXT (JSON) | Details (score, confidence) |
| timestamp | TIMESTAMP | |

### `model_metadata`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Always 1 |
| model_version | INTEGER | |
| last_trained | TIMESTAMP | |
| training_samples | INTEGER | |
| model_accuracy | REAL | |

## Packaging

Build a standalone executable with PyInstaller:

```bash
pyinstaller locksy.spec
```

The output will be in the `dist/` directory as `Locksy.exe` (Windows) or a macOS bundle.

### ⚠️ Antivirus Warning (Locksy.exe)

When packaged as a standalone `.exe`, Windows Defender and other antivirus software **will flag `Locksy.exe` as a keylogger/trojan**. This is a **false positive** caused by:

1. **pynput library**: The executable contains code that registers global keyboard and mouse hooks (`SetWindowsHookEx` / `GetMessage`). This is the same API used by malicious keyloggers, so AV engines flag it.
2. **PyInstaller bundling**: Packing Python + libraries into a single `.exe` triggers heuristic detection because the binary structure resembles known malware packers.

**What Locksy actually does:**
- Captures only **timing metadata** (key press/release timestamps) — never the content of what was typed
- All data stays **100% local** — no network transmission
- The behavior monitoring is used solely for **authentication**, not surveillance

**Mitigation:**
- Add an exception for `Locksy.exe` in your antivirus settings
- Or run from source with `python app.py` (no false positive since no PyInstaller bundle)
- Or sign the executable with a code signing certificate (reduces but may not eliminate flags)

We cannot whitelist the binary with AV vendors — you must set a local exclusion.

## Known Issues

1. **pysqlcipher3 build requirement**: Requires the SQLCipher C library to build. Falls back to plain SQLite if unavailable.
2. **Eel browser dependency**: Requires Chrome or Edge for the embedded web view.
3. **TensorFlow startup time**: First launch may be slow due to TF initialization.

## License

Provided as-is for educational and security research purposes.

---

**Last Updated**: June 2026 
**Version**: 2.0.0 (Desktop)

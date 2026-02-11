# AI-Continuous-Authentication-System

An AI-driven continuous authentication system that verifies users in real-time using behavioral biometrics such as keystroke dynamics, mouse movement patterns, scrolling behavior, and activity rhythms.

---

## 📌 Overview

Traditional authentication systems verify users only once during login.  
This project implements a **continuous authentication framework** that monitors user behavior throughout the session to detect unauthorized access or session hijacking.

The system uses machine learning–based anomaly detection to compare real-time behavioral data with a learned user profile.

---

## 🎯 Objectives

- Implement continuous post-login authentication
- Extract behavioral biometric features from user activity
- Train anomaly detection models for user verification
- Detect session hijacking in real-time
- Ensure privacy by avoiding sensitive content storage

---

## 🧠 System Workflow

1. **Data Collection**
   - Keystroke timing (dwell & flight time)
   - Mouse movement (x, y coordinates)
   - Scroll behavior
   - Idle time

2. **Feature Extraction**
   - Speed, acceleration, movement angle
   - Statistical window-based features
   - Activity rhythm patterns

3. **Model Training**
   - One-Class SVM
   - Isolation Forest
   - Autoencoder (optional)

4. **Continuous Verification**
   - Segment logs into time windows (5–10 sec)
   - Generate authenticity score
   - Detect anomalies

5. **Security Response**
   - Alert trigger
   - Secondary authentication
   - Session lock (if needed)

---

## 📊 Dataset Structure

The behavioral log dataset includes:

- `timestamp`
- `user_id`
- `session_id`
- `event_type`
- `key_category`
- `key_dwell`
- `mouse_x`
- `mouse_y`
- `mouse_event`
- `scroll_dx`
- `scroll_dy`
- `idle_seconds`

Raw logs are converted into feature vectors using time-window segmentation.

---

## 🛠 Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- (Optional) TensorFlow / PyTorch  

---

## 📂 Project Structure


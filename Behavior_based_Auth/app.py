#!/usr/bin/env python3
import eel
import sys
import os
import json
import threading
import time
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pynput import keyboard, mouse

from config import DesktopConfig
from database.db_manager import DatabaseManager
from utils.feature_extractor import BehavioralFeatureExtractor
from utils.drift_detector import BehavioralDriftDetector
from models.behavioral_models import EnsembleBehavioralClassifier
from utils.locks import lock_workstation
from utils.security import save_model_hashes, load_and_verify_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('behavioral_auth.log', mode='a')]
)
logger = logging.getLogger(__name__)

db = DatabaseManager(DesktopConfig.DATABASE_PATH, DesktopConfig.DATABASE_KEY)
extractor = BehavioralFeatureExtractor(window_size=DesktopConfig.WINDOW_SIZE)
drift_detector = BehavioralDriftDetector(
    window_size=DesktopConfig.DRIFT_DETECTION_WINDOW,
    alpha=DesktopConfig.DRIFT_ALPHA,
    min_samples=DesktopConfig.DRIFT_MIN_SAMPLES
)

ensemble: Optional[EnsembleBehavioralClassifier] = None
is_calibrated = False
monitoring_active = False

anomaly_event_count = 0
consecutive_anomalies = 0
soft_lock_active = False
soft_lock_verified = False
monitoring_paused_until: Optional[datetime] = None
tired_mode_until: Optional[datetime] = None

pynput_buffer = {'keystroke': deque(), 'mouse': deque()}
pynput_lock = threading.Lock()

calibration_features: Dict[str, list] = {'keystroke': [], 'mouse': []}

feature_history = deque(maxlen=500)

def start_pynput_listeners():
    key_state = {}

    def on_press(key_obj):
        try:
            key_str = str(key_obj)
            ts = time.time() * 1000
            key_state[key_str] = ts
            with pynput_lock:
                pynput_buffer['keystroke'].append({'key': key_str, 'type': 'keydown', 'timestamp': ts})
        except Exception:
            pass

    def on_release(key_obj):
        try:
            key_str = str(key_obj)
            ts = time.time() * 1000
            press_time = key_state.pop(key_str, ts)
            with pynput_lock:
                pynput_buffer['keystroke'].append({
                    'key': key_str, 'type': 'keyup', 'timestamp': ts,
                    'hold_time': ts - press_time, 'flight_time': 0
                })
        except Exception:
            pass

    def on_move(x, y):
        ts = time.time() * 1000
        with pynput_lock:
            pynput_buffer['mouse'].append({'x': x, 'y': y, 'type': 'move', 'timestamp': ts})

    def on_click(x, y, button, pressed):
        ts = time.time() * 1000
        btn = button.value if hasattr(button, 'value') else 0
        with pynput_lock:
            pynput_buffer['mouse'].append({
                'x': x, 'y': y, 'button': btn, 'type': 'click',
                'eventType': 'mousedown' if pressed else 'mouseup', 'timestamp': ts
            })

    def on_scroll(x, y, dx, dy):
        ts = time.time() * 1000
        with pynput_lock:
            pynput_buffer['mouse'].append({
                'x': x, 'y': y, 'type': 'scroll', 'dx': dx, 'dy': dy, 'timestamp': ts
            })

    kb = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    kb.daemon = True
    ms.daemon = True
    kb.start()
    ms.start()
    logger.info("pynput listeners started")

@eel.expose
def start_calibration_listeners():
    logger.info("Calibration UI signaled")

@eel.expose
def send_calibration_data(data_type: str, events: list):
    if data_type == 'keystroke':
        feat = extractor.extract_keystroke_features(events)
        if feat:
            calibration_features['keystroke'].append(feat)
    elif data_type == 'mouse':
        feat = extractor.extract_mouse_features(events)
        if feat:
            calibration_features['mouse'].append(feat)

@eel.expose
def complete_calibration() -> dict:
    global ensemble, is_calibrated
    try:
        ks_list = calibration_features['keystroke']
        m_list = calibration_features['mouse']
        if not ks_list and not m_list:
            return {'success': False, 'error': 'No calibration data collected'}
        all_combined = []
        max_len = max(len(ks_list), len(m_list))
        ks_padded = ks_list or [extractor._get_empty_keystroke_features()]
        m_padded = m_list or [extractor._get_empty_mouse_features()]
        ks_iter = (ks_padded[i % len(ks_padded)] for i in range(max_len))
        m_iter = (m_padded[i % len(m_padded)] for i in range(max_len))
        for ks_f, m_f in zip(ks_iter, m_iter):
            combined = extractor.get_combined_features(ks_f, m_f)
            all_combined.append(combined)
            feature_history.append(combined)
        ensemble = EnsembleBehavioralClassifier(1, DesktopConfig.MODELS_BASE_PATH)
        results = ensemble.train_initial_models(all_combined)
        ensemble.save_all_models()
        save_model_hashes(DesktopConfig.MODELS_BASE_PATH, DesktopConfig.DATABASE_KEY)
        db.set_calibrated(True)
        accuracy = results.get('gru', {}).get('accuracy', 0.8)
        db.update_model_metadata(accuracy=accuracy, training_samples=len(all_combined))
        is_calibrated = True
        logger.info(f"Calibration complete: {len(all_combined)} samples, accuracy={accuracy}")
        return {'success': True, 'accuracy': accuracy,
                'keystroke_samples': len(ks_list), 'mouse_samples': len(m_list)}
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return {'success': False, 'error': str(e)}

@eel.expose
def run_immediate_auth() -> dict:
    return perform_auth_check()

@eel.expose
def get_activity_log() -> list:
    log = []
    try:
        events = db.get_behavioral_data(limit=50)
        for ev in events:
            log.append({'time': ev.get('timestamp', ''), 'type': ev.get('data_type', 'event'),
                        'description': 'Feature data collected', 'risk': 'low'})
    except Exception:
        pass
    return log

@eel.expose
def set_local_pin(pin: str) -> dict:
    if not pin:
        db.clear_pin()
        logger.info("Local PIN cleared")
        return {'success': True}
    if len(pin) < 4:
        return {'success': False, 'error': 'PIN must be at least 4 characters'}
    db.set_pin(pin)
    logger.info("Local PIN set")
    return {'success': True}

@eel.expose
def has_local_pin() -> bool:
    return db.has_pin()

@eel.expose
def verify_pin(pin: str) -> bool:
    return db.verify_pin(pin)

@eel.expose
def dismiss_soft_lock() -> dict:
    global soft_lock_active, soft_lock_verified
    soft_lock_active = False
    soft_lock_verified = True
    logger.info("Soft lock dismissed after PIN verification")
    return {'success': True}

@eel.expose
def enable_tired_mode() -> dict:
    global tired_mode_until
    tired_mode_until = datetime.now() + timedelta(hours=DesktopConfig.TIRED_MODE_DURATION_HOURS)
    soft_lock_active = False
    logger.info(f"Tired mode enabled until {tired_mode_until}")
    db.log_event('tired_mode', {'until': tired_mode_until.isoformat()})
    return {'success': True, 'until': tired_mode_until.isoformat()}

@eel.expose
def enable_guest_mode(duration_minutes: int) -> dict:
    global monitoring_paused_until
    monitoring_paused_until = datetime.now() + timedelta(minutes=duration_minutes)
    soft_lock_active = False
    logger.info(f"Guest mode enabled for {duration_minutes} minutes")
    db.log_event('guest_mode', {'duration_minutes': duration_minutes,
                                 'until': monitoring_paused_until.isoformat()})
    return {'success': True, 'until': monitoring_paused_until.isoformat()}

@eel.expose
def get_lock_state() -> dict:
    now = datetime.now()
    result = {
        'soft_lock_active': soft_lock_active,
        'anomaly_event_count': anomaly_event_count,
        'has_pin': db.has_pin(),
    }
    if tired_mode_until and now < tired_mode_until:
        result['tired_mode_until'] = tired_mode_until.isoformat()
    else:
        result['tired_mode_until'] = None
    if monitoring_paused_until and now < monitoring_paused_until:
        result['monitoring_paused_until'] = monitoring_paused_until.isoformat()
    else:
        result['monitoring_paused_until'] = None
    return result

def get_effective_threshold() -> float:
    now = datetime.now()
    if tired_mode_until and now < tired_mode_until:
        return DesktopConfig.SOFT_LOCK_THRESHOLD_TIRED
    return DesktopConfig.DEFAULT_ANOMALY_THRESHOLD

def perform_auth_check() -> dict:
    global consecutive_anomalies, anomaly_event_count, soft_lock_active

    now = datetime.now()
    if monitoring_paused_until and now < monitoring_paused_until:
        return {'authenticity_score': 0.5, 'confidence': 0.0, 'anomaly_score': 0.0,
                'anomaly_detected': False, 'alert_level': 0,
                'alert_message': f'Monitoring paused until {monitoring_paused_until.isoformat()}',
                'consecutive_anomalies': 0, 'soft_lock_active': False}

    if not ensemble or not ensemble.gru_model.is_trained:
        return {'authenticity_score': 0.5, 'confidence': 0.0, 'anomaly_score': 0.0,
                'anomaly_detected': False, 'alert_level': 0, 'alert_message': 'Models not trained',
                'consecutive_anomalies': 0, 'soft_lock_active': False}

    try:
        ks_events = []
        mouse_events = []
        with pynput_lock:
            ks_events = list(pynput_buffer['keystroke'])
            mouse_events = list(pynput_buffer['mouse'])
            pynput_buffer['keystroke'].clear()
            pynput_buffer['mouse'].clear()

        if not ks_events and not mouse_events:
            if len(feature_history) < 5:
                return {'authenticity_score': 0.5, 'confidence': 0.0, 'anomaly_score': 0.0,
                        'anomaly_detected': False, 'alert_level': 0, 'alert_message': 'Insufficient data',
                        'consecutive_anomalies': 0, 'soft_lock_active': soft_lock_active}
            combined = feature_history[-1]
        else:
            ks_feat = extractor.extract_keystroke_features(ks_events) if ks_events else extractor._get_empty_keystroke_features()
            m_feat = extractor.extract_mouse_features(mouse_events) if mouse_events else extractor._get_empty_mouse_features()
            combined = extractor.get_combined_features(ks_feat, m_feat)
            feature_history.append(combined)

        if len(feature_history) < 5:
            return {'authenticity_score': 0.5, 'confidence': 0.0, 'anomaly_score': 0.0,
                    'anomaly_detected': False, 'alert_level': 0, 'alert_message': 'Building baseline',
                    'consecutive_anomalies': 0, 'soft_lock_active': soft_lock_active}

        prediction = ensemble.predict_ensemble(list(feature_history))
        auth_score = prediction['ensemble']['authenticity_score']
        confidence = prediction['ensemble']['confidence']
        anomaly_score = 1.0 - auth_score

        effective_threshold = get_effective_threshold()
        anomaly_detected = anomaly_score > effective_threshold

        if anomaly_detected:
            consecutive_anomalies += 1
            alert_msg = 'Normal behavior'
            alert_level = 0

            if consecutive_anomalies >= DesktopConfig.CONSECUTIVE_ANOMALIES_LIMIT:
                anomaly_event_count += 1

                if anomaly_event_count == 1:
                    alert_level = 3
                    alert_msg = 'First anomaly — hard locking workstation'
                    logger.warning(f"Anomaly event #{anomaly_event_count}: hard locking workstation")
                    db.log_event('hard_lock', {'anomaly_score': anomaly_score})
                    threading.Thread(target=lock_workstation, daemon=True).start()
                else:
                    alert_level = 4
                    alert_msg = 'Anomaly detected — soft lock activated'
                    soft_lock_active = True
                    logger.warning(f"Anomaly event #{anomaly_event_count}: soft lock activated")
                    db.log_event('soft_lock', {'anomaly_score': anomaly_score,
                                                'anomaly_event_count': anomaly_event_count})
            elif confidence > 0.7:
                alert_level = 2
                alert_msg = 'High confidence anomaly detected'
            else:
                alert_level = 1
                alert_msg = 'Low confidence anomaly detected'
        else:
            consecutive_anomalies = 0

        if anomaly_detected:
            db.log_event('anomaly', {'anomaly_score': anomaly_score, 'confidence': confidence,
                                      'auth_score': auth_score,
                                      'anomaly_event_count': anomaly_event_count})

        return {'authenticity_score': float(auth_score), 'confidence': float(confidence),
                'anomaly_score': float(anomaly_score), 'anomaly_detected': anomaly_detected,
                'alert_level': alert_level, 'alert_message': alert_msg,
                'consecutive_anomalies': consecutive_anomalies,
                'soft_lock_active': soft_lock_active,
                'anomaly_event_count': anomaly_event_count}

    except Exception as e:
        logger.error(f"Auth check error: {e}")
        return {'authenticity_score': 0.5, 'confidence': 0.0, 'anomaly_score': 0.5,
                'anomaly_detected': False, 'alert_level': 0, 'alert_message': f'Error: {e}',
                'consecutive_anomalies': 0, 'soft_lock_active': soft_lock_active}

def monitoring_loop():
    global monitoring_active
    monitoring_active = True
    while monitoring_active:
        try:
            now = datetime.now()
            if monitoring_paused_until and now >= monitoring_paused_until:
                monitoring_paused_until = None
                logger.info("Guest mode expired, monitoring resumed")

            if tired_mode_until and now >= tired_mode_until:
                tired_mode_until = None
                logger.info("Tired mode expired, normal threshold restored")
                db.log_event('tired_mode_expired', {})

            if is_calibrated and ensemble:
                result = perform_auth_check()
                eel.on_auth_update(result)

                if result['alert_level'] >= 2 and result['alert_level'] < 4:
                    eel.on_security_alert({'level': result['alert_level'],
                                           'message': result['alert_message'],
                                           'confidence': result['confidence']})

                if result['alert_level'] == 4:
                    eel.on_soft_lock({
                        'anomaly_event_count': result['anomaly_event_count'],
                        'anomaly_score': result['anomaly_score'],
                        'has_pin': db.has_pin(),
                    })

            time.sleep(DesktopConfig.FEATURE_UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Monitoring loop: {e}")
            time.sleep(DesktopConfig.FEATURE_UPDATE_INTERVAL)

def main():
    global is_calibrated, ensemble, anomaly_event_count

    is_calibrated = db.is_calibrated()
    if is_calibrated:
        try:
            if not load_and_verify_models(DesktopConfig.MODELS_BASE_PATH, DesktopConfig.DATABASE_KEY):
                logger.error("Model integrity check failed!")
                is_calibrated = False
            else:
                ensemble = EnsembleBehavioralClassifier(1, DesktopConfig.MODELS_BASE_PATH)
                ensemble.load_all_models()
                logger.info("Models loaded and verified")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            is_calibrated = False

    anomaly_event_count = 0
    start_pynput_listeners()

    monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor_thread.start()

    eel.init('web')
    start_page = 'calib.html' if not is_calibrated else 'challenge.html'
    kwargs = {'size': (1400, 900), 'port': DesktopConfig.EEL_PORT}
    if sys.platform == 'win32':
        kwargs['mode'] = 'chrome'
    try:
        eel.start(start_page, **kwargs)
    except Exception as e:
        logger.error(f"Eel start error: {e}, retrying without chrome mode")
        kwargs.pop('mode', None)
        eel.start(start_page, **kwargs)

if __name__ == '__main__':
    main()

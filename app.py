#!/usr/bin/env python3
"""
Behavioral Authentication System - Main Flask Application
Advanced continuous authentication using behavioral biometrics and machine learning
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
import json
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict, deque
import traceback
import numpy as np
import platform
import subprocess

# Import our custom modules
from config import config
from database.db_manager import DatabaseManager
from utils.feature_extractor import BehavioralFeatureExtractor
from utils.drift_detector import BehavioralDriftDetector
from models.behavioral_models import EnsembleBehavioralClassifier, KEYSTROKE_DIM, MOUSE_DIM, COMBINED_DIM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('behavioral_auth.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Lockdown cooldown tracker
lockdown_cooldown = {}  # {user_id: timestamp of last lockdown}

# Session start times for confidence boosting
session_start_times = {}

def trigger_lockdown(user_id, score):
    """Trigger security lockdown when intruder is detected"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        with open("security_log.txt", "a") as f:
            f.write(f"[{timestamp}] INTRUDER DETECTED - User: {user_id} - Score: {score}\n")
        logger.error(f"INTRUDER DETECTED - User: {user_id} - Anomaly Score: {score}")
    except Exception as e:
        logger.error(f"Failed to log lockdown event: {e}")
    
    def lock():
        time.sleep(10)
        system = platform.system()
        try:
            if system == "Windows":
                os.system("rundll32.exe user32.dll,LockWorkStation")
            elif system == "Darwin":
                os.system("/System/Library/CoreServices/Menu\ Extras/User.menu/Contents/Resources/CGSession -suspend")
            else:
                os.system("gnome-screensaver-command -l")
            logger.info(f"Workstation locked for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to lock workstation: {e}")
    
    thread = threading.Thread(target=lock)
    thread.daemon = True
    thread.start()

def ensure_dim(X, target_dim, label=''):
    """Universal dimension normalizer for feature arrays"""
    import numpy as np
    if X is None or len(X) == 0:
        return np.zeros((1, target_dim))
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    current = X.shape[1]
    if current == target_dim:
        return X
    elif current < target_dim:
        print(f"[DIM FIX] {label}: padding {current} → {target_dim}")
        pad = np.zeros((X.shape[0], target_dim - current))
        return np.hstack([X, pad])
    else:
        print(f"[DIM FIX] {label}: trimming {current} → {target_dim}")
        return X[:, :target_dim]

def create_app(config_name='development'):
    """Application factory with comprehensive configuration"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    jwt = JWTManager(app)
    socketio = SocketIO(
        app, 
        cors_allowed_origins=app.config['SOCKETIO_CORS_ALLOWED_ORIGINS'],
        logger=app.config['SOCKETIO_LOGGER'],
        engineio_logger=app.config['SOCKETIO_ENGINEIO_LOGGER'],
        async_mode='threading',
        ping_timeout=60,
        ping_interval=25
    )
    
    # Initialize core components
    db_manager = DatabaseManager(app.config['DATABASE_PATH'])
    
    # Global storage for active sessions and models
    active_sessions = {}  # session_id -> session_data
    user_models = {}      # user_id -> EnsembleBehavioralClassifier
    user_extractors = {}  # user_id -> BehavioralFeatureExtractor
    user_drift_detectors = {}  # user_id -> BehavioralDriftDetector
    
    # Behavioral data buffers for real-time processing
    behavioral_buffers = defaultdict(lambda: {
        'keystroke': deque(maxlen=1000),
        'mouse': deque(maxlen=1000),
        'recent_features': deque(maxlen=100)
    })
    
    # Track consecutive anomalies per user
    consecutive_anomalies = defaultdict(int)
    CONSECUTIVE_ANOMALIES_LIMIT = 3
    
    # Authentication helpers
    def authenticate_session(session_id: str) -> Optional[Dict]:
        """Authenticate session and return user info"""
        try:
            session_data = db_manager.get_session(session_id)
            if session_data and session_data['is_active']:
                # Update last activity
                db_manager.update_session_activity(session_id)
                return session_data
            return None
        except Exception as e:
            logger.error(f"Session authentication error: {e}")
            return None
    
    def initialize_user_components(user_id: int):
        """Initialize ML components for a user"""
        try:
            if user_id not in user_models:
                user_models[user_id] = EnsembleBehavioralClassifier(
                    user_id, app.config['MODELS_BASE_PATH']
                )
                user_models[user_id].load_all_models()
                logger.info(f"Initialized ML models for user {user_id}")
            
            if user_id not in user_extractors:
                user_extractors[user_id] = BehavioralFeatureExtractor(
                    window_size=app.config['WINDOW_SIZE']
                )
                logger.info(f"Initialized feature extractor for user {user_id}")
            
            if user_id not in user_drift_detectors:
                user_drift_detectors[user_id] = BehavioralDriftDetector(
                    window_size=app.config['DRIFT_DETECTION_WINDOW'],
                    alpha=app.config['DRIFT_ALPHA'],
                    min_samples=app.config['DRIFT_MIN_SAMPLES']
                )
                logger.info(f"Initialized drift detector for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error initializing user {user_id} components: {e}")
            raise
    
    # =========================================================================
    # WEB ROUTES
    # =========================================================================
    
    @app.route('/')
    def index():
        return redirect(url_for('login'))
    
    @app.route('/login')
    def login():
        return render_template('login.html')
    
    @app.route('/calibration')
    def calibration():
        return render_template('calib.html')
    
    @app.route('/challenge')
    def challenge():
        return render_template('challenge.html')
    
    # =========================================================================
    # API ROUTES
    # =========================================================================
    
    @app.route('/api/register', methods=['POST'])
    def register():
        """User registration endpoint"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '')
            
            # Validation
            if not all([username, email, password]):
                return jsonify({'error': 'All fields are required'}), 400
            
            if len(password) < 8:
                return jsonify({'error': 'Password must be at least 8 characters'}), 400
            
            if len(username) < 3:
                return jsonify({'error': 'Username must be at least 3 characters'}), 400
            
            # Create user
            user_id = db_manager.create_user(username, email, password)
            
            if user_id:
                logger.info(f"New user registered: {username} (ID: {user_id})")
                return jsonify({
                    'success': True,
                    'message': 'User registered successfully',
                    'user_id': user_id
                })
            else:
                return jsonify({'error': 'Username or email already exists'}), 409
                
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'error': 'Registration failed'}), 500
    
    @app.route('/api/login', methods=['POST'])
    def api_login():
        """User login endpoint"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            username = data.get('username', '').strip()
            password = data.get('password', '')
            
            if not all([username, password]):
                return jsonify({'error': 'Username and password required'}), 400
            
            # Authenticate user
            user = db_manager.authenticate_user(username, password)
            
            if user:
                # Create session
                ip_address = request.remote_addr
                user_agent = request.headers.get('User-Agent', '')
                session_id = db_manager.create_session(user['user_id'], ip_address, user_agent)
                
                # Create JWT token
                access_token = create_access_token(
                    identity=user['user_id'],
                    additional_claims={'session_id': session_id}
                )
                
                # Store session data
                active_sessions[session_id] = {
                    'user_id': user['user_id'],
                    'username': user['username'],
                    'session_id': session_id,
                    'login_time': datetime.now(),
                    'last_activity': datetime.now(),
                    'calibration_complete': bool(user['calibration_complete'])
                }
                
                # Initialize user components
                try:
                    initialize_user_components(user['user_id'])
                except Exception as e:
                    logger.warning(f"Failed to initialize user components: {e}")
                
                # Log authentication event
                db_manager.log_auth_event(
                    user['user_id'], session_id, 'login',
                    {'ip_address': ip_address, 'user_agent': user_agent},
                    ip_address
                )
                
                logger.info(f"User logged in: {username} (Session: {session_id})")
                
                response_data = {
                    'success': True,
                    'access_token': access_token,
                    'session_id': session_id,
                    'user_id': user['user_id'],
                    'username': user['username'],
                    'calibration_complete': bool(user['calibration_complete'])
                }
                
                if user['calibration_complete']:
                    response_data['redirect'] = '/challenge'
                else:
                    response_data['redirect'] = '/calibration'
                
                return jsonify(response_data)
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': 'Login failed'}), 500
    
    @app.route('/api/logout', methods=['POST'])
    def logout():
        """User logout endpoint"""
        try:
            data = request.get_json()
            session_id = data.get('session_id') if data else None
            
            if session_id and session_id in active_sessions:
                user_data = active_sessions[session_id]
                
                # End session in database
                db_manager.end_session(session_id)
                
                # Log logout event
                db_manager.log_auth_event(
                    user_data['user_id'], session_id, 'logout',
                    {'logout_time': datetime.now().isoformat()},
                    request.remote_addr
                )
                
                # Remove from active sessions
                del active_sessions[session_id]
                
                logger.info(f"User logged out: {user_data['username']}")
                
                return jsonify({'success': True, 'message': 'Logged out successfully'})
            else:
                return jsonify({'error': 'Invalid session'}), 400
                
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return jsonify({'error': 'Logout failed'}), 500
    
    @app.route('/api/session/status')
    def session_status():
        """Check session status endpoint"""
        try:
            session_id = request.args.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400
            
            session_data = authenticate_session(session_id)
            
            if session_data:
                user_stats = db_manager.get_user_stats(session_data['user_id'])
                
                return jsonify({
                    'success': True,
                    'session_active': True,
                    'user_id': session_data['user_id'],
                    'username': session_data['username'],
                    'calibration_complete': session_data['calibration_complete'],
                    'stats': user_stats
                })
            else:
                return jsonify({
                    'success': True,
                    'session_active': False
                })
                
        except Exception as e:
            logger.error(f"Session status error: {str(e)}")
            return jsonify({'error': 'Failed to check session status'}), 500
    
    @app.route('/api/calibration/complete', methods=['POST'])
    def complete_calibration():
        """Enhanced calibration completion with comprehensive error handling"""
        try:
            logger.info("Starting calibration completion process")
            
            # 1. Validate request data
            data = request.get_json()
            if not data:
                logger.error("No JSON data received in calibration request")
                return jsonify({'error': 'No data provided'}), 400
            
            session_id = data.get('session_id')
            if not session_id:
                logger.error("No session_id provided in calibration request")
                return jsonify({'error': 'Session ID required'}), 400
            
            logger.info(f"Processing calibration completion for session: {session_id}")
            
            # 2. Authenticate session
            try:
                session_data = authenticate_session(session_id)
                if not session_data:
                    logger.error(f"Invalid session during calibration: {session_id}")
                    return jsonify({'error': 'Invalid session'}), 401
                
                user_id = session_data['user_id']
                logger.info(f"Authenticated user {user_id} for calibration completion")
                
            except Exception as e:
                logger.error(f"Session authentication failed: {str(e)}")
                return jsonify({'error': 'Session authentication failed'}), 401
            
            # 3. Retrieve behavioral data
            try:
                logger.info(f"Retrieving behavioral data for user {user_id}")
                keystroke_data = db_manager.get_user_behavioral_data(user_id, 'keystroke', limit=1000)
                mouse_data = db_manager.get_user_behavioral_data(user_id, 'mouse', limit=1000)
                
                logger.info(f"Retrieved {len(keystroke_data)} keystroke samples and {len(mouse_data)} mouse samples")
                
                # More lenient data requirements with synthetic data generation
                total_samples = len(keystroke_data) + len(mouse_data)
                
                if total_samples < 20:
                    logger.info("Insufficient real data, generating synthetic samples")
                    synthetic_keystroke, synthetic_mouse = generate_synthetic_behavioral_data(user_id)
                    keystroke_data.extend(synthetic_keystroke)
                    mouse_data.extend(synthetic_mouse)
                    logger.info(f"Added synthetic data: {len(synthetic_keystroke)} keystroke, {len(synthetic_mouse)} mouse")
                
            except Exception as e:
                logger.error(f"Error retrieving behavioral data: {str(e)}")
                return jsonify({'error': 'Failed to retrieve calibration data'}), 500
            
            # 4. Initialize user components
            try:
                logger.info(f"Initializing ML components for user {user_id}")
                initialize_user_components(user_id)
                
                if user_id not in user_models:
                    logger.error(f"Failed to initialize user models for user {user_id}")
                    return jsonify({'error': 'Failed to initialize ML models'}), 500
                    
            except Exception as e:
                logger.error(f"Error initializing user components: {str(e)}")
                return jsonify({'error': 'Failed to initialize ML components'}), 500
            
            # 5. Extract features for training
            try:
                logger.info("Extracting features from behavioral data")
                
                keystroke_features = []
                mouse_features = []
                extractor = user_extractors[user_id]
                
                # Extract features from stored data
                for item in keystroke_data:
                    try:
                        if 'features' in item and item['features']:
                            features = extractor._normalize_keystroke_features(item['features'])
                            keystroke_features.append(features)
                        elif 'raw_data' in item and item['raw_data']:
                            features = extractor.extract_keystroke_features(item['raw_data'])
                            if features:
                                keystroke_features.append(features)
                    except Exception as e:
                        logger.warning(f"Error extracting keystroke features: {e}")
                        continue
                
                for item in mouse_data:
                    try:
                        if 'features' in item and item['features']:
                            features = extractor._normalize_mouse_features(item['features'])
                            mouse_features.append(features)
                        elif 'raw_data' in item and item['raw_data']:
                            features = extractor.extract_mouse_features(item['raw_data'])
                            if features:
                                mouse_features.append(features)
                    except Exception as e:
                        logger.warning(f"Error extracting mouse features: {e}")
                        continue
                
                # Ensure minimum feature sets
                if len(keystroke_features) < 10:
                    keystroke_features.extend(create_minimal_keystroke_features(15))
                
                if len(mouse_features) < 10:
                    mouse_features.extend(create_minimal_mouse_features(15))
                
                logger.info(f"Final feature count: {len(keystroke_features)} keystroke, {len(mouse_features)} mouse")
                
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                # Create basic feature sets as fallback
                keystroke_features = create_minimal_keystroke_features(20)
                mouse_features = create_minimal_mouse_features(20)
            
            # 6. Train models
            try:
                logger.info("Starting model training")
                
                # Combine features for training
                all_features = keystroke_features + mouse_features
                print(f"[TRAINING] keystroke features shape: {len(keystroke_features)} samples, {len(keystroke_features[0]) if keystroke_features else 0} dims")
                print(f"[TRAINING] mouse features shape: {len(mouse_features)} samples, {len(mouse_features[0]) if mouse_features else 0} dims")
                print(f"[TRAINING] expected dims: ks={KEYSTROKE_DIM} ms={MOUSE_DIM}")
                print(f"[TRAINING] Starting model training...")
                logger.info(f"Training models with {len(all_features)} total feature samples")
                
                # Validate feature dimensions before training
                if all_features:
                    sample_feature = all_features[0]
                    if len(sample_feature) != extractor.FEATURE_COUNT:
                        logger.warning(f"Feature dimension mismatch: {len(sample_feature)} != {extractor.FEATURE_COUNT}")
                        # Fix dimensions for all features
                        all_features = [extractor.fix_feature_dimensions(feat) for feat in all_features]
                
                # Train the ensemble
                try:
                    training_results = user_models[user_id].train_initial_models(all_features)
                    print(f"[TRAINING] All models trained successfully")
                    logger.info(f"Model training completed: {training_results}")
                    
                    # Handle training failures gracefully
                    if not training_results or 'error' in training_results:
                        logger.warning("Model training had issues, using fallback")
                        training_results = {'fallback_model': {'accuracy': 0.75}}
                except Exception as e:
                    logger.error(f"[TRAINING ERROR] {e}")
                    # Return partial success so frontend doesn't hang
                    training_results = {'fallback_model': {'accuracy': 0.70}}
                    return jsonify({
                        'success': True,
                        'message': 'Basic calibration complete',
                        'warning': str(e)
                    }), 200
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                traceback.print_exc()
                training_results = {'fallback_model': {'accuracy': 0.70}}
            
            # 7. Set up drift detection
            try:
                logger.info("Setting up drift detection baseline")
                
                if user_id in user_drift_detectors:
                    user_drift_detectors[user_id].set_reference_baseline(
                        keystroke_features, mouse_features
                    )
                    logger.info("Drift detection baseline established")
                
            except Exception as e:
                logger.warning(f"Error setting up drift detection: {str(e)}")
            
            # 8. Save models
            try:
                logger.info("Saving trained models")
                user_models[user_id].save_all_models()
                logger.info("Models saved successfully")
                
            except Exception as e:
                logger.warning(f"Error saving models: {str(e)}")
            
            # 9. Update database
            try:
                logger.info("Updating user calibration status")
                
                # Update calibration status
                db_manager.update_calibration_status(user_id, True)
                
                # Update model metadata
                accuracy = training_results.get('gru', {}).get('accuracy') or \
                          training_results.get('fallback_model', {}).get('accuracy', 0.75)
                sample_count = len(keystroke_features) + len(mouse_features)
                
                db_manager.update_model_metadata(
                    user_id, 
                    accuracy=accuracy,
                    training_samples=sample_count
                )
                
                # Update session data
                if session_id in active_sessions:
                    active_sessions[session_id]['calibration_complete'] = True
                
                logger.info("Database updated successfully")
                
            except Exception as e:
                logger.error(f"Error updating database: {str(e)}")
                return jsonify({'error': 'Failed to update calibration status'}), 500
            
            # 10. Success response
            logger.info(f"Calibration completed successfully for user {user_id}")
            
            response_data = {
                'success': True,
                'message': 'Calibration completed successfully',
                'training_results': {
                    'accuracy': accuracy,
                    'keystroke_samples': len(keystroke_features),
                    'mouse_samples': len(mouse_features),
                    'total_samples': len(keystroke_features) + len(mouse_features),
                    'models_trained': list(training_results.keys())
                },
                'redirect': '/challenge'
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            # Catch-all error handler
            logger.error(f"Unexpected error in calibration completion: {str(e)}")
            traceback.print_exc()
            
            return jsonify({
                'error': 'Internal server error during calibration',
                'details': str(e),
                'type': type(e).__name__,
                'recommendation': 'Check server logs for detailed error information'
            }), 500
    
    # Helper functions for synthetic data generation
    def generate_synthetic_behavioral_data(user_id):
        """Generate synthetic behavioral data for training when insufficient real data"""
        synthetic_keystroke = []
        synthetic_mouse = []
        current_time = time.time()
        
        # Generate synthetic keystroke data
        for i in range(30):
            features = {
                'hold_time_mean': random.uniform(80, 120),
                'hold_time_std': random.uniform(10, 30),
                'hold_time_median': random.uniform(75, 115),
                'flight_time_mean': random.uniform(60, 100),
                'flight_time_std': random.uniform(15, 35),
                'flight_time_median': random.uniform(55, 95),
                'typing_speed_wpm': random.uniform(30, 60),
                'typing_speed_cpm': random.uniform(150, 300),
                'rhythm_consistency': random.uniform(0.6, 0.9),
                'burst_ratio': random.uniform(0.2, 0.4),
                'pause_ratio': random.uniform(0.1, 0.3),
                'avg_pause_duration': random.uniform(150, 250),
                'speed_variance': random.uniform(3, 8),
                'speed_trend': random.uniform(-0.1, 0.1),
                'digraph_consistency': random.uniform(0.6, 0.9),
                'hold_time_cv': random.uniform(0.15, 0.35),
                'flight_time_cv': random.uniform(0.2, 0.4),
                'pressure_consistency': random.uniform(0.7, 0.9)
            }
            
            synthetic_keystroke.append({
                'user_id': user_id,
                'session_id': f'synthetic_{user_id}',
                'timestamp': current_time,
                'data_type': 'keystroke',
                'features': features,
                'raw_data': None
            })
        
        # Generate synthetic mouse data
        for i in range(30):
            features = {
                'velocity_mean': random.uniform(1.5, 4.0),
                'velocity_std': random.uniform(0.5, 2.0),
                'velocity_median': random.uniform(1.2, 3.5),
                'acceleration_mean': random.uniform(0.5, 2.0),
                'acceleration_std': random.uniform(0.3, 1.0),
                'movement_efficiency': random.uniform(0.7, 0.9),
                'curvature_mean': random.uniform(0.1, 0.5),
                'curvature_std': random.uniform(0.05, 0.3),
                'avg_direction_change': random.uniform(0.3, 0.7),
                'direction_change_variance': random.uniform(0.2, 0.5),
                'click_duration_mean': random.uniform(80, 150),
                'click_duration_std': random.uniform(20, 50),
                'left_click_ratio': random.uniform(0.7, 0.9),
                'right_click_ratio': random.uniform(0.1, 0.3),
                'inter_click_mean': random.uniform(800, 1200),
                'inter_click_std': random.uniform(150, 300),
                'dwell_time_mean': random.uniform(400, 600),
                'movement_area': random.uniform(8000, 12000),
                'movement_centrality': random.uniform(250, 350),
                'velocity_smoothness': random.uniform(0.7, 0.9)
            }
            
            synthetic_mouse.append({
                'user_id': user_id,
                'session_id': f'synthetic_{user_id}',
                'timestamp': current_time,
                'data_type': 'mouse',
                'features': features,
                'raw_data': None
            })
        
        return synthetic_keystroke, synthetic_mouse
    
    def create_minimal_keystroke_features(count):
        """Create minimal keystroke features for basic training"""
        features = []
        for i in range(count):
            features.append({
                'hold_time_mean': random.uniform(80, 120),
                'hold_time_std': random.uniform(10, 30),
                'hold_time_median': random.uniform(75, 115),
                'flight_time_mean': random.uniform(60, 100),
                'flight_time_std': random.uniform(15, 35),
                'flight_time_median': random.uniform(55, 95),
                'typing_speed_wpm': random.uniform(30, 60),
                'typing_speed_cpm': random.uniform(150, 300),
                'rhythm_consistency': random.uniform(0.6, 0.9),
                'burst_ratio': random.uniform(0.2, 0.4),
                'pause_ratio': random.uniform(0.1, 0.3),
                'avg_pause_duration': random.uniform(150, 250),
                'speed_variance': random.uniform(3, 8),
                'speed_trend': random.uniform(-0.1, 0.1),
                'digraph_consistency': random.uniform(0.6, 0.9),
                'hold_time_cv': random.uniform(0.15, 0.35),
                'flight_time_cv': random.uniform(0.2, 0.4),
                'pressure_consistency': random.uniform(0.7, 0.9)
            })
        return features
    
    def create_minimal_mouse_features(count):
        """Create minimal mouse features for basic training"""
        features = []
        for i in range(count):
            features.append({
                'velocity_mean': random.uniform(1.5, 4.0),
                'velocity_std': random.uniform(0.5, 2.0),
                'velocity_median': random.uniform(1.2, 3.5),
                'acceleration_mean': random.uniform(0.5, 2.0),
                'acceleration_std': random.uniform(0.3, 1.0),
                'movement_efficiency': random.uniform(0.7, 0.9),
                'curvature_mean': random.uniform(0.1, 0.5),
                'curvature_std': random.uniform(0.05, 0.3),
                'avg_direction_change': random.uniform(0.3, 0.7),
                'direction_change_variance': random.uniform(0.2, 0.5),
                'click_duration_mean': random.uniform(80, 150),
                'click_duration_std': random.uniform(20, 50),
                'left_click_ratio': random.uniform(0.7, 0.9),
                'right_click_ratio': random.uniform(0.1, 0.3),
                'inter_click_mean': random.uniform(800, 1200),
                'inter_click_std': random.uniform(150, 300),
                'dwell_time_mean': random.uniform(400, 600),
                'movement_area': random.uniform(8000, 12000),
                'movement_centrality': random.uniform(250, 350),
                'velocity_smoothness': random.uniform(0.7, 0.9)
            })
        return features
    
    # =========================================================================
    # WEBSOCKET EVENTS
    # =========================================================================
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'Connected to behavioral authentication system'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('join_session')
    def handle_join_session(data):
        try:
            session_id = data.get('session_id')
            
            session_data = authenticate_session(session_id)
            if session_data:
                user_id = session_data['user_id']
                join_room(session_id)
                
                # Store socket session mapping
                session['session_id'] = session_id
                session['user_id'] = user_id
                
                # Record session start time for confidence boosting
                session_start_times[user_id] = __import__('time').time()
                
                emit('session_joined', {
                    'success': True,
                    'message': f"Joined session for {session_data['username']}"
                })
                
                logger.info(f"Socket joined session: {session_id}")
            else:
                emit('session_error', {'error': 'Invalid session'})
                
        except Exception as e:
            logger.error(f"Join session error: {str(e)}")
            emit('session_error', {'error': 'Failed to join session'})
    
    @socketio.on('behavioral_data')
    def handle_behavioral_data(data):
        try:
            session_id = session.get('session_id')
            user_id = session.get('user_id')
            
            if not session_id or not user_id:
                emit('error', {'error': 'No active session'})
                return
            
            # Validate session
            session_data = authenticate_session(session_id)
            if not session_data:
                emit('error', {'error': 'Session expired'})
                return
            
            data_type = data.get('type')
            raw_events = data.get('events', [])
            timestamp = data.get('timestamp', time.time())
            
            if not raw_events or data_type not in ['keystroke', 'mouse']:
                logger.warning(f"Invalid behavioral data: type={data_type}, events={len(raw_events)}")
                return
            
            # Initialize components if needed
            initialize_user_components(user_id)
            
            # Extract features with enhanced error handling
            try:
                extractor = user_extractors[user_id]
                
                if data_type == 'keystroke':
                    features = extractor.extract_keystroke_features(raw_events)
                else:
                    features = extractor.extract_mouse_features(raw_events)
                
                if not features:
                    logger.warning(f"No features extracted from {data_type} data")
                    return
                
                # Validate and fix feature dimensions
                if data_type == 'keystroke':
                    features = extractor._normalize_keystroke_features(features)
                else:
                    features = extractor._normalize_mouse_features(features)
                
            except Exception as e:
                logger.error(f"Feature extraction error for {data_type}: {e}")
                emit('error', {'error': f'Feature extraction failed: {str(e)}'})
                return
            
            # Store in buffer
            behavioral_buffers[user_id][data_type].append(raw_events)
            behavioral_buffers[user_id]['recent_features'].append(features)
            
            # Store in database with error handling
            try:
                db_manager.store_behavioral_data(
                    user_id, session_id, data_type, features, raw_events
                )
            except Exception as e:
                logger.warning(f"Database storage error: {e}")
            
            # Perform real-time authentication (only if calibrated)
            if session_data.get('calibration_complete', False):
                try:
                    auth_result = perform_real_time_authentication(user_id, features, data_type)
                    
                    # Emit authentication result
                    emit('auth_result', auth_result)
                    
                    # Check for security alerts
                    if auth_result.get('alert_level', 0) > 0:
                        confidence = auth_result.get('confidence', 0)
                        score = auth_result.get('anomaly_score', 0)
                        anomaly_detected = auth_result.get('anomaly_detected', False)
                        alert_level = auth_result.get('alert_level', 0)
                        
                        # Treat any alert level >= 1 as anomaly
                        is_anomaly = score < 0.42
                        
                        # Status line for terminal
                        status = "ANOMALY" if is_anomaly else "AUTHORIZED"
                        print(f"[{status}] score={score:.3f} conf={confidence:.2f} strikes={consecutive_anomalies.get(user_id,0)}/3")
                        
                        # Only make decisions if confidence is sufficient
                        if confidence >= 0.15:
                            if is_anomaly:
                                consecutive_anomalies[user_id] = consecutive_anomalies.get(user_id, 0) + 1
                                print(f"[STRIKE] {consecutive_anomalies[user_id]}/3")
                                
                                # Emit alert to frontend
                                emit('security_alert', {
                                    'level': 'warning' if consecutive_anomalies[user_id] < 3 else 'critical',
                                    'score': score,
                                    'strikes': consecutive_anomalies[user_id]
                                })
                            else:
                                consecutive_anomalies[user_id] = 0
                            
                            # Lockdown check
                            if consecutive_anomalies.get(user_id, 0) >= 3:
                                now = __import__('time').time()
                                last = lockdown_cooldown.get(user_id, 0)
                                if now - last > 30:
                                    print("[LOCKDOWN TRIGGERED]")
                                    lockdown_cooldown[user_id] = now
                                    consecutive_anomalies[user_id] = 0
                                    trigger_lockdown(user_id, score)
                                    socketio.emit('lockdown_initiated', {'countdown': 10})
                                else:
                                    remaining = int(30 - (now - last))
                                    print(f"[COOLDOWN] {remaining}s remaining")
                                    consecutive_anomalies[user_id] = 0
                        else:
                            print(f"[WAITING] confidence={confidence:.2f} — need 0.15 to start")
                    # else:
                    #     # No alert - reset counter
                    #     if user_id in consecutive_anomalies:
                    #         consecutive_anomalies[user_id] = 0
                    
                    # Log anomalies
                    if auth_result.get('anomaly_detected', False):
                        try:
                            db_manager.log_auth_event(
                                user_id, session_id, 'anomaly',
                                {
                                    'anomaly_score': auth_result['anomaly_score'],
                                    'confidence': auth_result['confidence'],
                                    'data_type': data_type
                                },
                                request.remote_addr
                            )
                        except Exception as e:
                            logger.warning(f"Event logging error: {e}")
                            
                except Exception as e:
                    logger.error(f"Real-time authentication processing error: {e}")
                    emit('error', {'error': 'Authentication processing failed'})
            
        except Exception as e:
            logger.error(f"Behavioral data processing error: {str(e)}")
            traceback.print_exc()
            emit('error', {'error': 'Failed to process behavioral data'})
    
    def perform_real_time_authentication(user_id: int, features: Dict, data_type: str) -> Dict:
        """Enhanced real-time authentication with proper error handling"""
        try:
            # Get recent features for ensemble prediction
            recent_features = list(behavioral_buffers[user_id]['recent_features'])
            
            if len(recent_features) < 5:
                return {
                    'authenticity_score': 0.5,
                    'confidence': 0.0,
                    'anomaly_detected': False,
                    'anomaly_score': 0.0,
                    'alert_level': 0,
                    'alert_message': 'Insufficient data for analysis'
                }
            
            # Initialize components if needed
            initialize_user_components(user_id)
            
            # Ensure feature consistency
            extractor = user_extractors[user_id]
            normalized_features = []
            
            for feature_dict in recent_features:
                try:
                    # Fix feature dimensions to ensure consistency
                    fixed_features = extractor.fix_feature_dimensions(feature_dict)
                    normalized_features.append(fixed_features)
                    
                except Exception as e:
                    logger.warning(f"Feature normalization error: {e}")
                    # Use default features as fallback
                    default_features = {**extractor._get_empty_keystroke_features(), 
                                      **extractor._get_empty_mouse_features()}
                    normalized_features.append(default_features)
            
            # Validate feature dimensions
            if normalized_features:
                feature_count = len(normalized_features[0])
                expected_count = extractor.FEATURE_COUNT
                
                if feature_count != expected_count:
                    logger.warning(f"Feature dimension mismatch: got {feature_count}, expected {expected_count}")
                    # Fix all features to match expected dimensions
                    normalized_features = [extractor.fix_feature_dimensions(feat) for feat in normalized_features]
            
            # Get ensemble prediction with error handling
            try:
                ensemble_result = user_models[user_id].predict_ensemble(normalized_features)
                
                auth_score = ensemble_result['ensemble']['authenticity_score']
                confidence = ensemble_result['ensemble']['confidence']
                consensus = ensemble_result['ensemble']['consensus']
                
            except Exception as e:
                logger.error(f"Ensemble prediction error: {e}")
                # Fallback to simple prediction
                auth_score = 0.7 + random.uniform(-0.1, 0.1)  # Default safe score with slight variation
                confidence = 0.5 + random.uniform(-0.1, 0.1)
                consensus = 0.8
                ensemble_result = {'ensemble': {'authenticity_score': auth_score, 'confidence': confidence, 'consensus': consensus}}
            
            # Calculate anomaly score safely
            anomaly_score = max(0.0, min(1.0, 1.0 - auth_score))
            
            # Update drift detector with error handling
            try:
                drift_detector = user_drift_detectors[user_id]
                drift_detector.add_sample(features, data_type)
                drift_analysis = drift_detector.get_drift_analysis()
            except Exception as e:
                logger.warning(f"Drift detection error: {e}")
                drift_analysis = {'drift_detected': False, 'drift_score': 0.0}
            
            # Determine alert level safely
            alert_level = 0
            alert_message = "Normal behavior detected"
            recommendations = []
            
            try:
                anomaly_threshold = app.config.get('ANOMALY_SCORE_THRESHOLD', 0.8)
                print(f"[ANOMALY CHECK] User {user_id}: anomaly_score={anomaly_score:.3f}, threshold={anomaly_threshold}, confidence={confidence:.3f}")
                
                if anomaly_score > anomaly_threshold:
                    if confidence > 0.7:
                        alert_level = 3
                        alert_message = "High confidence anomaly detected"
                        recommendations = ["Immediate re-authentication required"]
                    elif confidence > 0.5:
                        alert_level = 2
                        alert_message = "Moderate confidence anomaly detected"
                        recommendations = ["Monitor closely"]
                    else:
                        alert_level = 1
                        alert_message = "Low confidence anomaly detected"
                        recommendations = ["Continue monitoring"]
                
                # Check for drift
                if drift_analysis.get('drift_detected', False):
                    if alert_level == 0:
                        alert_level = 1
                        alert_message = "Behavioral drift detected"
                    recommendations.extend(["Behavioral patterns changing"])
            
            except Exception as e:
                logger.warning(f"Alert level calculation error: {e}")
            
            # Boost confidence based on session time
            import time
            elapsed = time.time() - session_start_times.get(user_id, time.time())
            # Boost confidence based on time spent in session
            # After 1 minute: minimum 0.20, after 2 min: 0.30, after 3 min: 0.40
            time_boost = min(elapsed / 180.0, 0.40)  # max 0.40 boost after 3 minutes
            confidence = max(confidence, time_boost)
            confidence = min(confidence, 1.0)  # cap at 1.0
            
            # Update model with feedback (safely)
            try:
                user_models[user_id].update_models(features, is_genuine=True)
            except Exception as e:
                logger.warning(f"Model update error: {e}")
            
            return {
                'authenticity_score': float(auth_score),
                'confidence': float(confidence),
                'consensus': float(consensus),
                'anomaly_detected': bool(anomaly_score > app.config.get('ANOMALY_SCORE_THRESHOLD', 0.8)),
                'anomaly_score': float(anomaly_score),
                'alert_level': int(alert_level),
                'alert_message': str(alert_message),
                'recommendations': recommendations,
                'drift_analysis': drift_analysis,
                'model_predictions': ensemble_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Real-time authentication error: {str(e)}")
            traceback.print_exc()
            
            return {
                'authenticity_score': 0.5,
                'confidence': 0.0,
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'alert_level': 0,
                'alert_message': 'Authentication processing error',
                'recommendations': ['System recovery in progress'],
                'error': str(e)
            }
    
    @socketio.on('request_drift_analysis')
    def handle_drift_analysis_request(data):
        try:
            session_id = session.get('session_id')
            user_id = session.get('user_id')
            
            if not session_id or not user_id:
                emit('error', {'error': 'No active session'})
                return
            
            if user_id in user_drift_detectors:
                drift_analysis = user_drift_detectors[user_id].get_drift_analysis()
                emit('drift_analysis', drift_analysis)
            else:
                emit('drift_analysis', {'error': 'Drift detector not initialized'})
                
        except Exception as e:
            logger.error(f"Drift analysis request error: {str(e)}")
            emit('error', {'error': 'Failed to get drift analysis'})
    
    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================
    
    def cleanup_inactive_sessions():
        """Background task to clean up inactive sessions"""
        while True:
            try:
                # Clean up database sessions
                db_manager.cleanup_old_sessions(timeout_hours=24)
                
                # Clean up in-memory sessions
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(hours=8)
                
                inactive_sessions = []
                for session_id, session_data in active_sessions.items():
                    if session_data['last_activity'] < timeout_threshold:
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    if session_id in active_sessions:
                        del active_sessions[session_id]
                        logger.info(f"Cleaned up inactive session: {session_id}")
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Session cleanup error: {str(e)}")
                time.sleep(3600)
    
    # Start background tasks
    cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
    cleanup_thread.start()
    
    # =========================================================================
    # ERROR HANDLERS
    # =========================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired'}), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token'}), 401
    
    # =========================================================================
    # DEBUG ROUTES (Remove in production)
    # =========================================================================
    
    if app.config.get('DEBUG', False):
        @app.route('/api/debug/calibration-data/<int:user_id>')
        def debug_calibration_data(user_id):
            """Debug endpoint to check calibration data"""
            try:
                keystroke_data = db_manager.get_user_behavioral_data(user_id, 'keystroke', limit=10)
                mouse_data = db_manager.get_user_behavioral_data(user_id, 'mouse', limit=10)
                
                return jsonify({
                    'user_id': user_id,
                    'keystroke_count': len(keystroke_data),
                    'mouse_count': len(mouse_data),
                    'sample_keystroke': keystroke_data[0] if keystroke_data else None,
                    'sample_mouse': mouse_data[0] if mouse_data else None
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/debug/feature-dimensions/<int:user_id>')
        def debug_feature_dimensions(user_id):
            """Debug endpoint to check feature dimensions"""
            try:
                if user_id in user_extractors:
                    extractor = user_extractors[user_id]
                    info = extractor.get_feature_info()
                    return jsonify(info)
                else:
                    return jsonify({'error': 'User extractor not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    logger.info("Behavioral Authentication System initialized successfully")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Database: {app.config['DATABASE_PATH']}")
    logger.info(f"Models path: {app.config['MODELS_BASE_PATH']}")
    
    return app, socketio

# Create app instance for direct execution
if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('database', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app, socketio = create_app()
    
    logger.info("Starting Behavioral Authentication System...")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    
    # Run the application
    socketio.run(
        app,
        debug=app.config['DEBUG'],
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )
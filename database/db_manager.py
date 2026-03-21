import sqlite3
import json
import hashlib
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
from contextlib import contextmanager

class DatabaseManager:
    """Manages all database operations for the continuous authentication system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    calibration_complete BOOLEAN DEFAULT 0
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Behavioral data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavioral_data (
                    data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_type TEXT NOT NULL,  -- 'keystroke' or 'mouse'
                    features TEXT NOT NULL,   -- JSON string of features
                    raw_data TEXT,           -- JSON string of raw measurements
                    confidence_score REAL,
                    anomaly_score REAL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Authentication events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT,
                    event_type TEXT NOT NULL,  -- 'login', 'logout', 'anomaly', 'drift'
                    event_data TEXT,          -- JSON string of event details
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Model metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    user_id INTEGER PRIMARY KEY,
                    model_version INTEGER DEFAULT 1,
                    last_trained TIMESTAMP,
                    training_samples INTEGER DEFAULT 0,
                    model_accuracy REAL,
                    drift_detected BOOLEAN DEFAULT 0,
                    drift_timestamp TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Create a new user"""
        try:
            # Generate salt and hash password
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, salt)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash, salt))
                
                user_id = cursor.lastrowid
                
                # Initialize model metadata
                cursor.execute('''
                    INSERT INTO model_metadata (user_id, last_trained)
                    VALUES (?, ?)
                ''', (user_id, datetime.now()))
                
                conn.commit()
                return user_id
                
        except sqlite3.IntegrityError:
            return None  # User already exists
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM users WHERE username = ? AND is_active = 1
            ''', (username,))
            
            user = cursor.fetchone()
            if not user:
                return None
            
            # Check if account is locked
            if user['locked_until'] and datetime.fromisoformat(user['locked_until']) > datetime.now():
                return None
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
                # Reset failed attempts and update last login
                cursor.execute('''
                    UPDATE users SET failed_attempts = 0, last_login = ?, locked_until = NULL
                    WHERE user_id = ?
                ''', (datetime.now(), user['user_id']))
                conn.commit()
                
                return dict(user)
            else:
                # Increment failed attempts
                failed_attempts = user['failed_attempts'] + 1
                locked_until = None
                
                if failed_attempts >= 5:  # Max attempts from config
                    locked_until = datetime.now() + timedelta(minutes=15)
                
                cursor.execute('''
                    UPDATE users SET failed_attempts = ?, locked_until = ?
                    WHERE user_id = ?
                ''', (failed_attempts, locked_until, user['user_id']))
                conn.commit()
                
                return None
    
    def create_session(self, user_id: int, ip_address: str, user_agent: str) -> str:
        """Create a new session"""
        session_id = hashlib.sha256(f"{user_id}{datetime.now()}{ip_address}".encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, user_id, ip_address, user_agent)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_id, ip_address, user_agent))
            conn.commit()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, u.username, u.calibration_complete
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.session_id = ? AND s.is_active = 1
            ''', (session_id,))
            
            session = cursor.fetchone()
            return dict(session) if session else None
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET last_activity = ?
                WHERE session_id = ? AND is_active = 1
            ''', (datetime.now(), session_id))
            conn.commit()
    
    def end_session(self, session_id: str):
        """End an active session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = 0
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
    
    def store_behavioral_data(self, user_id: int, session_id: str, data_type: str, 
                            features: Dict, raw_data: Dict = None, 
                            confidence_score: float = None, anomaly_score: float = None):
        """Store behavioral biometric data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO behavioral_data 
                (user_id, session_id, data_type, features, raw_data, confidence_score, anomaly_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, data_type, 
                  json.dumps(features), 
                  json.dumps(raw_data) if raw_data else None,
                  confidence_score, anomaly_score))
            conn.commit()
    
    def get_user_behavioral_data(self, user_id: int, data_type: str = None, 
                               limit: int = 1000) -> List[Dict]:
        """Get behavioral data for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM behavioral_data 
                WHERE user_id = ?
            '''
            params = [user_id]
            
            if data_type:
                query += ' AND data_type = ?'
                params.append(data_type)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['features'] = json.loads(row_dict['features'])
                if row_dict['raw_data']:
                    row_dict['raw_data'] = json.loads(row_dict['raw_data'])
                results.append(row_dict)
            
            return results
    
    def log_auth_event(self, user_id: int, session_id: str, event_type: str, 
                      event_data: Dict, ip_address: str = None):
        """Log authentication events"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO auth_events (user_id, session_id, event_type, event_data, ip_address)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_id, event_type, json.dumps(event_data), ip_address))
            conn.commit()
    
    def update_calibration_status(self, user_id: int, is_complete: bool):
        """Update user calibration status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET calibration_complete = ?
                WHERE user_id = ?
            ''', (is_complete, user_id))
            conn.commit()
    
    def update_model_metadata(self, user_id: int, accuracy: float = None, 
                            training_samples: int = None, drift_detected: bool = None):
        """Update model metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if accuracy is not None:
                updates.append('model_accuracy = ?')
                params.append(accuracy)
                updates.append('last_trained = ?')
                params.append(datetime.now())
            
            if training_samples is not None:
                updates.append('training_samples = ?')
                params.append(training_samples)
            
            if drift_detected is not None:
                updates.append('drift_detected = ?')
                params.append(drift_detected)
                if drift_detected:
                    updates.append('drift_timestamp = ?')
                    params.append(datetime.now())
            
            if updates:
                params.append(user_id)
                cursor.execute(f'''
                    UPDATE model_metadata SET {', '.join(updates)}
                    WHERE user_id = ?
                ''', params)
                conn.commit()
    
    def get_model_metadata(self, user_id: int) -> Optional[Dict]:
        """Get model metadata for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_metadata WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def cleanup_old_sessions(self, timeout_hours: int = 24):
        """Clean up old inactive sessions"""
        cutoff_time = datetime.now() - timedelta(hours=timeout_hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = 0
                WHERE last_activity < ? AND is_active = 1
            ''', (cutoff_time,))
            conn.commit()
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get behavioral data count
            cursor.execute('''
                SELECT COUNT(*) as total_samples,
                       COUNT(CASE WHEN data_type = 'keystroke' THEN 1 END) as keystroke_samples,
                       COUNT(CASE WHEN data_type = 'mouse' THEN 1 END) as mouse_samples
                FROM behavioral_data WHERE user_id = ?
            ''', (user_id,))
            data_stats = dict(cursor.fetchone())
            
            # Get session count
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_sessions
                FROM sessions WHERE user_id = ?
            ''', (user_id,))
            session_stats = dict(cursor.fetchone())
            
            # Get recent anomalies
            cursor.execute('''
                SELECT COUNT(*) as recent_anomalies
                FROM auth_events 
                WHERE user_id = ? AND event_type = 'anomaly' 
                AND timestamp > datetime('now', '-7 days')
            ''', (user_id,))
            anomaly_stats = dict(cursor.fetchone())
            
            return {**data_stats, **session_stats, **anomaly_stats}
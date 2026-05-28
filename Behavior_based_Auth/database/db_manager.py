import sqlite3
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

try:
    import pysqlcipher3.dbapi2 as sqlcipher
    HAS_SQLCIPHER = True
    logger.info("pysqlcipher3 available — database will be encrypted")
except ImportError:
    HAS_SQLCIPHER = False
    logger.warning("pysqlcipher3 not installed — database will NOT be encrypted")

class DatabaseManager:
    def __init__(self, db_path: str, encryption_key: str = None):
        self.db_path = db_path
        self.encryption_key = encryption_key
        self.init_database()

    def _apply_cipher(self, conn):
        if self.encryption_key and HAS_SQLCIPHER:
            conn.execute(f"PRAGMA key = '{self.encryption_key}'")
        elif self.encryption_key and not HAS_SQLCIPHER:
            logger.warning("Encryption key configured but pysqlcipher3 not available — storing unencrypted")

    def init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                username TEXT DEFAULT 'User',
                calibration_complete INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS behavioral_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_type TEXT NOT NULL,
                features TEXT NOT NULL,
                raw_data TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS auth_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model_version INTEGER DEFAULT 1,
                last_trained TIMESTAMP,
                training_samples INTEGER DEFAULT 0,
                model_accuracy REAL
            )''')
            c.execute('''INSERT OR IGNORE INTO profile (id, username) VALUES (1, 'User')''')
            c.execute('''INSERT OR IGNORE INTO model_metadata (id) VALUES (1)''')
            conn.commit()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._apply_cipher(conn)
        try:
            yield conn
        finally:
            conn.close()

    def get_profile(self) -> Dict:
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM profile WHERE id = 1')
            row = c.fetchone()
            return dict(row) if row else {'calibration_complete': False}

    def set_profile(self, username: str = None):
        with self.get_connection() as conn:
            c = conn.cursor()
            if username:
                c.execute('UPDATE profile SET username = ? WHERE id = 1', (username,))
            conn.commit()

    def is_calibrated(self) -> bool:
        prof = self.get_profile()
        return bool(prof.get('calibration_complete', False))

    def set_calibrated(self, status: bool):
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('UPDATE profile SET calibration_complete = ? WHERE id = 1', (int(status),))
            conn.commit()

    def store_behavioral_data(self, data_type: str, features: Dict, raw_data: list = None):
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO behavioral_data (data_type, features, raw_data)
                         VALUES (?, ?, ?)''',
                      (data_type, json.dumps(features), json.dumps(raw_data) if raw_data else None))
            conn.commit()

    def get_behavioral_data(self, data_type: str = None, limit: int = 1000) -> List[Dict]:
        with self.get_connection() as conn:
            c = conn.cursor()
            q = 'SELECT * FROM behavioral_data'
            params = []
            if data_type:
                q += ' WHERE data_type = ?'
                params.append(data_type)
            q += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            c.execute(q, params)
            results = []
            for row in c.fetchall():
                d = dict(row)
                d['features'] = json.loads(d['features'])
                if d['raw_data']:
                    d['raw_data'] = json.loads(d['raw_data'])
                results.append(d)
            return results

    def log_event(self, event_type: str, event_data: Dict = None):
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO auth_events (event_type, event_data)
                         VALUES (?, ?)''', (event_type, json.dumps(event_data) if event_data else None))
            conn.commit()

    def update_model_metadata(self, accuracy: float = None, training_samples: int = None):
        with self.get_connection() as conn:
            c = conn.cursor()
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
            if updates:
                params.append(1)
                c.execute(f'UPDATE model_metadata SET {", ".join(updates)} WHERE id = ?', params)
                conn.commit()

    def get_model_metadata(self) -> Optional[Dict]:
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM model_metadata WHERE id = 1')
            row = c.fetchone()
            return dict(row) if row else None

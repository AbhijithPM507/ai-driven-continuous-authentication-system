import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class GRUSequenceModel:
    """GRU model for sequential behavioral data analysis"""
    
    def __init__(self, sequence_length: int = 50, feature_dim: int = 38, 
                 hidden_units: int = 64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None  # Store feature names for consistency
        
    def build_model(self):
        """Build the GRU model architecture"""
        model = Sequential([
            Input(shape=(self.sequence_length, self.feature_dim)),
            GRU(self.hidden_units, return_sequences=True, dropout=0.2),
            GRU(self.hidden_units // 2, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names
            self.feature_names = sorted(set().union(*(f.keys() for f in features)))
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        for f in features:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features
    
    def prepare_sequences(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to sequences"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        # Convert to matrix
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        # Normalize features
        if not self.is_trained:
            data_matrix = self.scaler.fit_transform(data_matrix)
        else:
            data_matrix = self.scaler.transform(data_matrix)
        
        # Create sequences
        sequences = []
        for i in range(len(data_matrix) - self.sequence_length + 1):
            sequences.append(data_matrix[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train(self, genuine_features: List[Dict], 
              imposter_features: List[Dict] = None) -> Dict:
        """Train the GRU model"""
        # Prepare genuine data
        genuine_sequences = self.prepare_sequences(genuine_features)
        
        if len(genuine_sequences) == 0:
            raise ValueError("Insufficient data for training")
        
        # Create labels (1 for genuine, 0 for imposter)
        genuine_labels = np.ones(len(genuine_sequences))
        
        X = genuine_sequences
        y = genuine_labels
        
        # Add imposter data if available
        if imposter_features:
            imposter_sequences = self.prepare_sequences(imposter_features)
            if len(imposter_sequences) > 0:
                imposter_labels = np.zeros(len(imposter_sequences))
                X = np.concatenate([genuine_sequences, imposter_sequences])
                y = np.concatenate([genuine_labels, imposter_labels])
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Train model
        early_stopping = EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        predictions = self.model.predict(X)
        binary_preds = (predictions > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': float(accuracy_score(y, binary_preds)),
            'precision': float(precision_score(y, binary_preds, zero_division=0)),
            'recall': float(recall_score(y, binary_preds, zero_division=0)),
            'loss': float(history.history['loss'][-1])
        }
        
        return metrics
    
    def predict(self, features: List[Dict]) -> Tuple[float, float]:
        """Predict authenticity score and confidence"""
        if not self.is_trained or self.model is None:
            return 0.5, 0.0
        
        sequences = self.prepare_sequences(features)
        if len(sequences) == 0:
            return 0.5, 0.0
        
        # Get prediction for the most recent sequence
        prediction = self.model.predict(sequences[-1:], verbose=0)[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to confidence scale
        
        return float(prediction), float(confidence)
    
    def save(self, filepath: str):
        """Save the model and scaler"""
        if self.model is not None:
            self.model.save(f"{filepath}_gru.h5")
            joblib.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f"{filepath}_gru_scaler.pkl")
    
    def load(self, filepath: str):
        """Load the model and scaler"""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}_gru.h5")
            data = joblib.load(f"{filepath}_gru_scaler.pkl")
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.is_trained = True
            return True
        except:
            return False

class AutoencoderAnomalyDetector:
    """Autoencoder for detecting behavioral anomalies"""
    
    def __init__(self, feature_dim: int = 38, encoding_dim: int = 8):
        self.feature_dim = feature_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        self.is_trained = False
        self.feature_names = None
        
    def build_model(self):
        """Build the autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=(self.feature_dim,))
        encoded = Dense(self.encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(self.encoding_dim * 2, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(self.feature_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = autoencoder
        return autoencoder
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names
            self.feature_names = sorted(set().union(*(f.keys() for f in features)))
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        for f in features:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features
    
    def prepare_data(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to matrix"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        if not self.is_trained:
            data_matrix = self.scaler.fit_transform(data_matrix)
        else:
            data_matrix = self.scaler.transform(data_matrix)
        
        return data_matrix
    
    def train(self, genuine_features: List[Dict]) -> Dict:
        """Train the autoencoder on genuine user data"""
        X = self.prepare_data(genuine_features)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        # Update feature_dim based on actual features
        self.feature_dim = len(self.feature_names)
        
        if self.model is None:
            self.build_model()
        
        # Train autoencoder
        early_stopping = EarlyStopping(
            monitor='loss', patience=15, restore_best_weights=True
        )
        
        history = self.model.fit(
            X, X,  # Autoencoder learns to reconstruct input
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction errors on training data
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(reconstruction_errors, 95)
        self.is_trained = True
        
        metrics = {
            'loss': float(history.history['loss'][-1]),
            'threshold': float(self.threshold),
            'mean_reconstruction_error': float(np.mean(reconstruction_errors))
        }
        
        return metrics
    
    def predict_anomaly_score(self, features: List[Dict]) -> float:
        """Predict anomaly score for new data"""
        if not self.is_trained or self.model is None:
            return 0.5
        
        X = self.prepare_data(features)
        if len(X) == 0:
            return 0.5
        
        # Calculate reconstruction error for most recent data
        reconstruction = self.model.predict(X[-1:], verbose=0)
        error = np.mean(np.square(X[-1:] - reconstruction))
        
        # Normalize error to [0, 1] scale
        if self.threshold > 0:
            anomaly_score = min(error / self.threshold, 1.0)
        else:
            anomaly_score = 0.0
        
        return float(anomaly_score)
    
    def save(self, filepath: str):
        """Save the model and scaler"""
        if self.model is not None:
            self.model.save(f"{filepath}_autoencoder.h5")
            joblib.dump({
                'scaler': self.scaler,
                'threshold': self.threshold,
                'feature_names': self.feature_names
            }, f"{filepath}_autoencoder_params.pkl")
    
    def load(self, filepath: str):
        """Load the model and scaler"""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}_autoencoder.h5")
            params = joblib.load(f"{filepath}_autoencoder_params.pkl")
            self.scaler = params['scaler']
            self.threshold = params['threshold']
            self.feature_names = params['feature_names']
            if self.feature_names:
                self.feature_dim = len(self.feature_names)
            self.is_trained = True
            return True
        except:
            return False

class OneClassSVMDetector:
    """One-Class SVM for outlier detection"""
    
    def __init__(self, nu: float = 0.1, gamma: str = 'scale'):
        self.model = OneClassSVM(nu=nu, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names
            self.feature_names = sorted(set().union(*(f.keys() for f in features)))
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        for f in features:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features
        
    def prepare_data(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to matrix"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        if not self.is_trained:
            data_matrix = self.scaler.fit_transform(data_matrix)
        else:
            data_matrix = self.scaler.transform(data_matrix)
        
        return data_matrix
    
    def train(self, genuine_features: List[Dict]) -> Dict:
        """Train One-Class SVM on genuine data"""
        X = self.prepare_data(genuine_features)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        self.model.fit(X)
        self.is_trained = True
        
        # Calculate training metrics
        predictions = self.model.predict(X)
        inlier_ratio = np.sum(predictions == 1) / len(predictions)
        
        return {
            'inlier_ratio': float(inlier_ratio),
            'support_vectors': int(len(self.model.support_vectors_))
        }
    
    def predict_outlier_score(self, features: List[Dict]) -> float:
        """Predict outlier score (higher = more likely outlier)"""
        if not self.is_trained:
            return 0.5
        
        X = self.prepare_data(features)
        if len(X) == 0:
            return 0.5
        
        # Get decision score for most recent data
        decision_score = self.model.decision_function(X[-1:])
        
        # Convert to [0, 1] scale (0 = outlier, 1 = inlier)
        # SVM decision scores are typically in range [-2, 2]
        normalized_score = (decision_score[0] + 2) / 4
        outlier_score = 1 - max(0, min(1, normalized_score))
        
        return float(outlier_score)
    
    def save(self, filepath: str):
        """Save the model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, f"{filepath}_svm.pkl")
    
    def load(self, filepath: str):
        """Load the model and scaler"""
        try:
            data = joblib.load(f"{filepath}_svm.pkl")
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.feature_names = data.get('feature_names', None)
            return True
        except:
            return False

class IncrementalKNNClassifier:
    """Incremental k-NN with sliding window for adaptive learning"""
    
    def __init__(self, k: int = 5, window_size: int = 1000):
        self.k = k
        self.window_size = window_size
        self.genuine_buffer = deque(maxlen=window_size)
        self.imposter_buffer = deque(maxlen=window_size // 4)  # Smaller imposter buffer
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names from all available features
            if isinstance(features, list):
                self.feature_names = sorted(set().union(*(f.keys() for f in features)))
            else:
                self.feature_names = sorted(features.keys())
                features = [features]
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        feature_list = features if isinstance(features, list) else [features]
        for f in feature_list:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features if isinstance(features, list) else consistent_features[0]
        
    def prepare_data(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to matrix"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        return data_matrix
    
    def update(self, features: Dict, is_genuine: bool):
        """Update the classifier with new data"""
        # Ensure feature consistency for single feature dict
        features = self._ensure_feature_consistency(features)
        
        if is_genuine:
            self.genuine_buffer.append(features)
        else:
            self.imposter_buffer.append(features)
        
        # Update scaler periodically
        if len(self.genuine_buffer) > 50 and len(self.genuine_buffer) % 20 == 0:
            all_data = list(self.genuine_buffer) + list(self.imposter_buffer)
            if all_data:
                data_matrix = self.prepare_data(all_data)
                self.scaler.fit(data_matrix)
                self.is_trained = True
    
    def predict(self, features: Dict) -> Tuple[float, float]:
        """Predict authenticity using k-NN"""
        if not self.is_trained or len(self.genuine_buffer) < self.k:
            return 0.5, 0.0
        
        # Ensure feature consistency for single feature dict
        features = self._ensure_feature_consistency(features)
        
        # Prepare query point
        query_data = self.prepare_data([features])
        if len(query_data) == 0:
            return 0.5, 0.0
        
        query_data = self.scaler.transform(query_data)
        query_point = query_data[0]
        
        # Prepare training data
        genuine_data = self.prepare_data(list(self.genuine_buffer))
        imposter_data = self.prepare_data(list(self.imposter_buffer))
        
        all_data = []
        all_labels = []
        
        if len(genuine_data) > 0:
            genuine_data = self.scaler.transform(genuine_data)
            all_data.extend(genuine_data)
            all_labels.extend([1] * len(genuine_data))
        
        if len(imposter_data) > 0:
            imposter_data = self.scaler.transform(imposter_data)
            all_data.extend(imposter_data)
            all_labels.extend([0] * len(imposter_data))
        
        if len(all_data) < self.k:
            return 0.5, 0.0
        
        # Find k nearest neighbors
        all_data = np.array(all_data)
        distances = np.linalg.norm(all_data - query_point, axis=1)
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # Calculate prediction
        k_nearest_labels = [all_labels[i] for i in k_nearest_indices]
        genuine_votes = sum(k_nearest_labels)
        confidence = abs(genuine_votes / self.k - 0.5) * 2
        
        return float(genuine_votes / self.k), float(confidence)
    
    def save(self, filepath: str):
        """Save the classifier state"""
        joblib.dump({
            'genuine_buffer': list(self.genuine_buffer),
            'imposter_buffer': list(self.imposter_buffer),
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, f"{filepath}_knn.pkl")
    
    def load(self, filepath: str):
        """Load the classifier state"""
        try:
            data = joblib.load(f"{filepath}_knn.pkl")
            self.genuine_buffer = deque(data['genuine_buffer'], maxlen=self.window_size)
            self.imposter_buffer = deque(data['imposter_buffer'], maxlen=self.window_size // 4)
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.feature_names = data.get('feature_names', None)
            return True
        except:
            return False

class PassiveAggressiveDetector:
    """Passive-Aggressive classifier for online learning"""
    
    def __init__(self, C: float = 1.0):
        self.model = PassiveAggressiveClassifier(C=C, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.sample_count = 0
        self.feature_names = None
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names
            self.feature_names = sorted(set().union(*(f.keys() for f in features)))
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        for f in features:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features
        
    def prepare_data(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to matrix"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        return data_matrix
    
    def partial_fit(self, features: List[Dict], labels: List[int]):
        """Update model with new data"""
        if not features:
            return
        
        X = self.prepare_data(features)
        y = np.array(labels)
        
        if not self.is_trained:
            # Initial fit with scaling
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y, classes=[0, 1])
            self.is_trained = True
        else:
            # Incremental update
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)
        
        self.sample_count += len(features)
    
    def predict(self, features: List[Dict]) -> Tuple[float, float]:
        """Predict authenticity probability"""
        if not self.is_trained:
            return 0.5, 0.0
        
        X = self.prepare_data(features)
        if len(X) == 0:
            return 0.5, 0.0
        
        X_scaled = self.scaler.transform(X)
        
        # Get prediction for most recent sample
        prediction = self.model.predict(X_scaled[-1:])[0]
        
        # Try to get prediction probability
        try:
            if hasattr(self.model, 'decision_function'):
                decision_score = self.model.decision_function(X_scaled[-1:])[0]
                # Convert decision score to probability-like score
                probability = 1 / (1 + np.exp(-decision_score))
                confidence = abs(probability - 0.5) * 2
            else:
                probability = float(prediction)
                confidence = 1.0 if prediction in [0, 1] else 0.0
        except:
            probability = float(prediction)
            confidence = 1.0 if prediction in [0, 1] else 0.0
        
        return float(probability), float(confidence)
    
    def save(self, filepath: str):
        """Save the model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'sample_count': self.sample_count,
            'feature_names': self.feature_names
        }, f"{filepath}_pa.pkl")
    
    def load(self, filepath: str):
        """Load the model and scaler"""
        try:
            data = joblib.load(f"{filepath}_pa.pkl")
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.sample_count = data['sample_count']
            self.feature_names = data.get('feature_names', None)
            return True
        except:
            return False

class IsolationForestDetector:
    """Isolation Forest for anomaly detection"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
    
    def _ensure_feature_consistency(self, features: List[Dict]) -> List[Dict]:
        """Ensure all feature dictionaries have the same keys"""
        if not features:
            return features
            
        if self.feature_names is None:
            # First time - establish feature names
            self.feature_names = sorted(set().union(*(f.keys() for f in features)))
        
        # Ensure all features have the same keys, filling missing with 0
        consistent_features = []
        for f in features:
            consistent_f = {name: f.get(name, 0.0) for name in self.feature_names}
            consistent_features.append(consistent_f)
        
        return consistent_features
        
    def prepare_data(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to matrix"""
        if not features:
            return np.array([])
        
        # Ensure feature consistency
        features = self._ensure_feature_consistency(features)
        
        data_matrix = np.array([[f[name] for name in self.feature_names] 
                               for f in features])
        
        if not self.is_trained:
            data_matrix = self.scaler.fit_transform(data_matrix)
        else:
            data_matrix = self.scaler.transform(data_matrix)
        
        return data_matrix
    
    def train(self, genuine_features: List[Dict]) -> Dict:
        """Train Isolation Forest on genuine data"""
        X = self.prepare_data(genuine_features)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        self.model.fit(X)
        self.is_trained = True
        
        # Calculate training metrics
        predictions = self.model.predict(X)
        inlier_ratio = np.sum(predictions == 1) / len(predictions)
        
        return {
            'inlier_ratio': float(inlier_ratio),
            'n_estimators': self.model.n_estimators
        }
    
    def predict_anomaly_score(self, features: List[Dict]) -> float:
        """Predict anomaly score"""
        if not self.is_trained:
            return 0.5
        
        X = self.prepare_data(features)
        if len(X) == 0:
            return 0.5
        
        # Get anomaly score for most recent data
        anomaly_score = self.model.decision_function(X[-1:])
        
        # Convert to [0, 1] scale (0 = normal, 1 = anomaly)
        # Isolation Forest scores are typically in range [-1, 1]
        normalized_score = (anomaly_score[0] + 1) / 2
        anomaly_score = 1 - max(0, min(1, normalized_score))
        
        return float(anomaly_score)
    
    def save(self, filepath: str):
        """Save the model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, f"{filepath}_isolation.pkl")
    
    def load(self, filepath: str):
        """Load the model and scaler"""
        try:
            data = joblib.load(f"{filepath}_isolation.pkl")
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.feature_names = data.get('feature_names', None)
            return True
        except:
            return False

class EnsembleBehavioralClassifier:
    """Ensemble of all behavioral models for robust authentication"""
    
    def __init__(self, user_id: int, models_path: str):
        self.user_id = user_id
        self.models_path = os.path.join(models_path, str(user_id))
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize all models
        self.gru_model = GRUSequenceModel()
        self.autoencoder = AutoencoderAnomalyDetector()
        self.svm_detector = OneClassSVMDetector()
        self.knn_classifier = IncrementalKNNClassifier()
        self.pa_classifier = PassiveAggressiveDetector()
        self.isolation_forest = IsolationForestDetector()
        
        self.model_weights = {
            'gru': 0.25,
            'autoencoder': 0.15,
            'svm': 0.15,
            'knn': 0.20,
            'pa': 0.15,
            'isolation': 0.10
        }
        
    def train_initial_models(self, genuine_features: List[Dict]) -> Dict:
        """Train all models on initial genuine data"""
        results = {}
        
        try:
            # Train GRU
            if len(genuine_features) >= 50:  # Need enough data for sequences
                results['gru'] = self.gru_model.train(genuine_features)
            
            # Train Autoencoder
            if len(genuine_features) >= 20:
                results['autoencoder'] = self.autoencoder.train(genuine_features)
            
            # Train One-Class SVM
            if len(genuine_features) >= 10:
                results['svm'] = self.svm_detector.train(genuine_features)
            
            # Initialize k-NN with genuine data
            for features in genuine_features:
                self.knn_classifier.update(features, is_genuine=True)
            
            # Train Isolation Forest
            if len(genuine_features) >= 20:
                results['isolation'] = self.isolation_forest.train(genuine_features)
            
            # Initialize Passive-Aggressive with genuine data
            if len(genuine_features) >= 5:
                labels = [1] * len(genuine_features)
                self.pa_classifier.partial_fit(genuine_features, labels)
                results['pa'] = {'initialized': True}
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def predict_ensemble(self, features: List[Dict]) -> Dict:
        """Get ensemble prediction from all models"""
        predictions = {}
        
        # GRU prediction
        try:
            gru_score, gru_conf = self.gru_model.predict(features)
            predictions['gru'] = {'score': gru_score, 'confidence': gru_conf}
        except:
            predictions['gru'] = {'score': 0.5, 'confidence': 0.0}
        
        # Autoencoder anomaly score
        try:
            ae_anomaly = self.autoencoder.predict_anomaly_score(features)
            predictions['autoencoder'] = {'anomaly_score': ae_anomaly}
        except:
            predictions['autoencoder'] = {'anomaly_score': 0.5}
        
        # SVM outlier score
        try:
            svm_outlier = self.svm_detector.predict_outlier_score(features)
            predictions['svm'] = {'outlier_score': svm_outlier}
        except:
            predictions['svm'] = {'outlier_score': 0.5}
        
        # k-NN prediction
        try:
            knn_score, knn_conf = self.knn_classifier.predict(features[-1])
            predictions['knn'] = {'score': knn_score, 'confidence': knn_conf}
        except:
            predictions['knn'] = {'score': 0.5, 'confidence': 0.0}
        
        # Passive-Aggressive prediction
        try:
            pa_score, pa_conf = self.pa_classifier.predict(features)
            predictions['pa'] = {'score': pa_score, 'confidence': pa_conf}
        except:
            predictions['pa'] = {'score': 0.5, 'confidence': 0.0}
        
        # Isolation Forest anomaly score
        try:
            if_anomaly = self.isolation_forest.predict_anomaly_score(features)
            predictions['isolation'] = {'anomaly_score': if_anomaly}
        except:
            predictions['isolation'] = {'anomaly_score': 0.5}
        
        # Calculate weighted ensemble score
        ensemble_score = self._calculate_ensemble_score(predictions)
        
        predictions['ensemble'] = ensemble_score
        return predictions
    
    def _calculate_ensemble_score(self, predictions: Dict) -> Dict:
        """Calculate weighted ensemble authentication score"""
        scores = []
        weights = []
        
        # GRU score
        if 'gru' in predictions and predictions['gru']['confidence'] > 0.1:
            scores.append(predictions['gru']['score'])
            weights.append(self.model_weights['gru'] * predictions['gru']['confidence'])
        
        # Autoencoder (convert anomaly to authenticity)
        if 'autoencoder' in predictions:
            auth_score = 1 - predictions['autoencoder']['anomaly_score']
            scores.append(auth_score)
            weights.append(self.model_weights['autoencoder'])
        
        # SVM (convert outlier to authenticity)
        if 'svm' in predictions:
            auth_score = 1 - predictions['svm']['outlier_score']
            scores.append(auth_score)
            weights.append(self.model_weights['svm'])
        
        # k-NN score
        if 'knn' in predictions and predictions['knn']['confidence'] > 0.1:
            scores.append(predictions['knn']['score'])
            weights.append(self.model_weights['knn'] * predictions['knn']['confidence'])
        
        # Passive-Aggressive score
        if 'pa' in predictions and predictions['pa']['confidence'] > 0.1:
            scores.append(predictions['pa']['score'])
            weights.append(self.model_weights['pa'] * predictions['pa']['confidence'])
        
        # Isolation Forest (convert anomaly to authenticity)
        if 'isolation' in predictions:
            auth_score = 1 - predictions['isolation']['anomaly_score']
            scores.append(auth_score)
            weights.append(self.model_weights['isolation'])
        
        # Calculate weighted average
        if scores and weights:
            weighted_score = np.average(scores, weights=weights)
            confidence = np.mean([abs(s - 0.5) * 2 for s in scores])
            consensus = np.std(scores)  # Lower std = higher consensus
        else:
            weighted_score = 0.5
            confidence = 0.0
            consensus = 1.0
        
        return {
            'authenticity_score': float(weighted_score),
            'confidence': float(confidence),
            'consensus': float(1 - min(consensus, 1.0)),  # Convert to consensus score
            'num_models': len(scores)
        }
    
    def update_models(self, features: Dict, is_genuine: bool):
        """Update models with new feedback"""
        # Update incremental models
        self.knn_classifier.update(features, is_genuine)
        
        # Update Passive-Aggressive
        label = 1 if is_genuine else 0
        self.pa_classifier.partial_fit([features], [label])
    
    def save_all_models(self):
        """Save all trained models"""
        base_path = os.path.join(self.models_path, 'model')
        
        self.gru_model.save(base_path)
        self.autoencoder.save(base_path)
        self.svm_detector.save(base_path)
        self.knn_classifier.save(base_path)
        self.pa_classifier.save(base_path)
        self.isolation_forest.save(base_path)
    
    def load_all_models(self):
        """Load all saved models"""
        base_path = os.path.join(self.models_path, 'model')
        
        results = {
            'gru': self.gru_model.load(base_path),
            'autoencoder': self.autoencoder.load(base_path),
            'svm': self.svm_detector.load(base_path),
            'knn': self.knn_classifier.load(base_path),
            'pa': self.pa_classifier.load(base_path),
            'isolation': self.isolation_forest.load(base_path)
        }
        
        return results
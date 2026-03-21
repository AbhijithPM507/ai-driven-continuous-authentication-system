import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import statistics
import math
import logging

logger = logging.getLogger(__name__)

class BehavioralFeatureExtractor:
    """
    Enhanced behavioral biometric feature extractor with consistent feature dimensions.
    Extracts 38 total features: 18 keystroke + 20 mouse features.
    """
    
    # keystroke features: 18
    # mouse features: 20
    # combined total: 38
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.keystroke_buffer = deque(maxlen=1000)
        self.mouse_buffer = deque(maxlen=2000)
        
        # FIXED FEATURE DEFINITIONS - DO NOT MODIFY ORDER OR COUNT
        # These ensure exactly 38 features total for ML model consistency
        
        # Keystroke features (18 total)
        self.KEYSTROKE_FEATURES = [
            'hold_time_mean', 'hold_time_std', 'hold_time_median',           # 3
            'flight_time_mean', 'flight_time_std', 'flight_time_median',     # 3
            'typing_speed_wpm', 'typing_speed_cpm',                          # 2
            'rhythm_consistency', 'burst_ratio', 'pause_ratio',              # 3
            'avg_pause_duration', 'speed_variance', 'speed_trend',           # 3
            'digraph_consistency', 'hold_time_cv', 'flight_time_cv',         # 3
            'pressure_consistency'                                            # 1
        ]  # Total: 18 features
        
        # Mouse features (20 total)
        self.MOUSE_FEATURES = [
            'velocity_mean', 'velocity_std', 'velocity_median',              # 3
            'acceleration_mean', 'acceleration_std',                         # 2
            'movement_efficiency', 'curvature_mean', 'curvature_std',        # 3
            'avg_direction_change', 'direction_change_variance',             # 2
            'click_duration_mean', 'click_duration_std',                     # 2
            'left_click_ratio', 'right_click_ratio',                         # 2
            'inter_click_mean', 'inter_click_std',                           # 2
            'dwell_time_mean', 'movement_area',                              # 2
            'movement_centrality', 'velocity_smoothness'                     # 2
        ]  # Total: 20 features
        
        # Combined feature list (38 total)
        self.ALL_FEATURES = self.KEYSTROKE_FEATURES + self.MOUSE_FEATURES
        self.FEATURE_COUNT = len(self.ALL_FEATURES)  # Should always be 38
        
        assert self.FEATURE_COUNT == 38, f"Feature count mismatch! Got {self.FEATURE_COUNT}, expected 38"
        
        logger.info(f"BehavioralFeatureExtractor initialized with {self.FEATURE_COUNT} features")
        
    def extract_keystroke_features(self, keystroke_data: List[Dict]) -> Dict:
        """
        Extract keystroke dynamics features with guaranteed consistent dimensions.
        
        Args:
            keystroke_data: List of keystroke events with timing information
            
        Returns:
            Dict with exactly 18 keystroke features
        """
        if len(keystroke_data) < 2:
            return self._get_empty_keystroke_features()
        
        try:
            features = {}
            
            # Extract timing data with multiple possible key names
            hold_times = []
            flight_times = []
            timestamps = []
            
            for event in keystroke_data:
                # Handle different possible key names for timing data
                hold_time = event.get('hold_time') or event.get('holdTime') or event.get('duration')
                flight_time = event.get('flight_time') or event.get('flightTime') or event.get('interval')
                timestamp = event.get('timestamp') or event.get('time')
                
                if hold_time is not None and isinstance(hold_time, (int, float)) and hold_time > 0:
                    hold_times.append(hold_time)
                
                if flight_time is not None and isinstance(flight_time, (int, float)) and flight_time > 0:
                    flight_times.append(flight_time)
                    
                if timestamp is not None:
                    timestamps.append(timestamp)
            
            # Basic timing features
            if hold_times:
                features.update(self._extract_timing_stats('hold_time', hold_times))
            
            if flight_times:
                features.update(self._extract_timing_stats('flight_time', flight_times))
            
            # Speed and rhythm features
            features.update(self._extract_speed_features(keystroke_data, timestamps))
            features.update(self._extract_rhythm_features(keystroke_data, timestamps))
            
            # Digraph and consistency features
            features.update(self._extract_consistency_features(keystroke_data))
            
            # Pressure features (if available)
            features.update(self._extract_pressure_features(keystroke_data))
            
            # Ensure exactly 18 features are returned
            return self._normalize_keystroke_features(features)
            
        except Exception as e:
            logger.warning(f"Keystroke feature extraction error: {e}")
            return self._get_empty_keystroke_features()
    
    def extract_mouse_features(self, mouse_data: List[Dict]) -> Dict:
        """
        Extract mouse behavior features with guaranteed consistent dimensions.
        
        Args:
            mouse_data: List of mouse events with position and timing information
            
        Returns:
            Dict with exactly 20 mouse features
        """
        if len(mouse_data) < 2:
            return self._get_empty_mouse_features()
        
        try:
            features = {}
            
            # Extract movement features
            features.update(self._extract_movement_features(mouse_data))
            
            # Extract click features
            features.update(self._extract_click_features(mouse_data))
            
            # Extract trajectory features
            features.update(self._extract_trajectory_features(mouse_data))
            
            # Extract behavioral patterns
            features.update(self._extract_behavioral_patterns(mouse_data))
            
            # Ensure exactly 20 features are returned
            return self._normalize_mouse_features(features)
            
        except Exception as e:
            logger.warning(f"Mouse feature extraction error: {e}")
            return self._get_empty_mouse_features()
    
    def _extract_timing_stats(self, prefix: str, times: List[float]) -> Dict:
        """Extract statistical features from timing data"""
        if not times or len(times) < 2:
            return {
                f'{prefix}_mean': 0.0,
                f'{prefix}_std': 0.0,
                f'{prefix}_median': 0.0
            }
        
        try:
            times_array = np.array(times)
            # Remove outliers (values beyond 3 standard deviations)
            mean = np.mean(times_array)
            std = np.std(times_array)
            if std > 0:
                times_array = times_array[np.abs(times_array - mean) <= 3 * std]
            
            if len(times_array) == 0:
                times_array = np.array(times)
            
            stats = {}
            stats[f'{prefix}_mean'] = float(np.mean(times_array))
            stats[f'{prefix}_std'] = float(np.std(times_array)) if len(times_array) > 1 else 0.0
            stats[f'{prefix}_median'] = float(np.median(times_array))
            
            return stats
            
        except Exception as e:
            logger.warning(f"Timing stats extraction error for {prefix}: {e}")
            return {
                f'{prefix}_mean': 0.0,
                f'{prefix}_std': 0.0,
                f'{prefix}_median': 0.0
            }
    
    def _extract_speed_features(self, keystroke_data: List[Dict], timestamps: List[float]) -> Dict:
        """Extract typing speed features"""
        features = {}
        
        try:
            if len(keystroke_data) < 10 or len(timestamps) < 2:
                features['typing_speed_wpm'] = 0.0
                features['typing_speed_cpm'] = 0.0
                features['speed_variance'] = 0.0
                features['speed_trend'] = 0.0
                return features
            
            # Calculate overall typing speed
            time_span = max(timestamps) - min(timestamps)
            if time_span > 0:
                chars_per_second = len(keystroke_data) / (time_span / 1000)  # Assuming timestamps in ms
                features['typing_speed_cpm'] = chars_per_second * 60
                features['typing_speed_wpm'] = features['typing_speed_cpm'] / 5  # Assuming 5 chars per word
            else:
                features['typing_speed_cpm'] = 0.0
                features['typing_speed_wpm'] = 0.0
            
            # Calculate speed variance using sliding windows
            window_size = min(10, len(keystroke_data) // 3)
            if window_size > 1:
                speeds = []
                for i in range(0, len(keystroke_data) - window_size + 1, window_size):
                    window_data = keystroke_data[i:i + window_size]
                    window_timestamps = [event.get('timestamp', 0) for event in window_data]
                    window_time = max(window_timestamps) - min(window_timestamps)
                    if window_time > 0:
                        speeds.append(len(window_data) / (window_time / 1000))
                
                if len(speeds) > 1:
                    features['speed_variance'] = float(np.var(speeds))
                    features['speed_trend'] = self._calculate_trend(speeds)
                else:
                    features['speed_variance'] = 0.0
                    features['speed_trend'] = 0.0
            else:
                features['speed_variance'] = 0.0
                features['speed_trend'] = 0.0
                
        except Exception as e:
            logger.warning(f"Speed features extraction error: {e}")
            features['typing_speed_wpm'] = 0.0
            features['typing_speed_cpm'] = 0.0
            features['speed_variance'] = 0.0
            features['speed_trend'] = 0.0
        
        return features
    
    def _extract_rhythm_features(self, keystroke_data: List[Dict], timestamps: List[float]) -> Dict:
        """Extract typing rhythm features"""
        features = {}
        
        try:
            if len(timestamps) < 3:
                features['rhythm_consistency'] = 0.0
                features['burst_ratio'] = 0.0
                features['pause_ratio'] = 0.0
                features['avg_pause_duration'] = 0.0
                return features
            
            # Calculate inter-key intervals
            inter_key_intervals = []
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                if interval > 0:
                    inter_key_intervals.append(interval)
            
            if len(inter_key_intervals) > 0:
                # Rhythm consistency (inverse of coefficient of variation)
                mean_interval = np.mean(inter_key_intervals)
                std_interval = np.std(inter_key_intervals)
                if mean_interval > 0:
                    features['rhythm_consistency'] = 1.0 / (1.0 + (std_interval / mean_interval))
                else:
                    features['rhythm_consistency'] = 0.0
                
                # Burst and pause analysis
                intervals_array = np.array(inter_key_intervals)
                burst_threshold = np.percentile(intervals_array, 25)
                pause_threshold = np.percentile(intervals_array, 75)
                
                bursts = intervals_array < burst_threshold
                pauses = intervals_array > pause_threshold
                
                features['burst_ratio'] = float(np.sum(bursts) / len(bursts))
                features['pause_ratio'] = float(np.sum(pauses) / len(pauses))
                
                if np.sum(pauses) > 0:
                    pause_durations = intervals_array[pauses]
                    features['avg_pause_duration'] = float(np.mean(pause_durations))
                else:
                    features['avg_pause_duration'] = 0.0
            else:
                features['rhythm_consistency'] = 0.0
                features['burst_ratio'] = 0.0
                features['pause_ratio'] = 0.0
                features['avg_pause_duration'] = 0.0
                
        except Exception as e:
            logger.warning(f"Rhythm features extraction error: {e}")
            features['rhythm_consistency'] = 0.0
            features['burst_ratio'] = 0.0
            features['pause_ratio'] = 0.0
            features['avg_pause_duration'] = 0.0
        
        return features
    
    def _extract_consistency_features(self, keystroke_data: List[Dict]) -> Dict:
        """Extract consistency and coefficient of variation features"""
        features = {}
        
        try:
            # Extract timing data for CV calculation
            hold_times = []
            flight_times = []
            
            for event in keystroke_data:
                hold_time = event.get('hold_time') or event.get('holdTime')
                flight_time = event.get('flight_time') or event.get('flightTime')
                
                if hold_time is not None and hold_time > 0:
                    hold_times.append(hold_time)
                if flight_time is not None and flight_time > 0:
                    flight_times.append(flight_time)
            
            # Calculate coefficient of variation for hold times
            if len(hold_times) > 1:
                mean_hold = np.mean(hold_times)
                std_hold = np.std(hold_times)
                features['hold_time_cv'] = std_hold / mean_hold if mean_hold > 0 else 0.0
            else:
                features['hold_time_cv'] = 0.0
            
            # Calculate coefficient of variation for flight times
            if len(flight_times) > 1:
                mean_flight = np.mean(flight_times)
                std_flight = np.std(flight_times)
                features['flight_time_cv'] = std_flight / mean_flight if mean_flight > 0 else 0.0
            else:
                features['flight_time_cv'] = 0.0
            
            # Digraph consistency (simplified)
            digraphs = defaultdict(list)
            for i in range(len(keystroke_data) - 1):
                key1 = keystroke_data[i].get('key', '')
                key2 = keystroke_data[i + 1].get('key', '')
                timing = keystroke_data[i + 1].get('flight_time', 0)
                
                if key1 and key2 and timing > 0:
                    digraphs[f"{key1}{key2}"].append(timing)
            
            if digraphs:
                all_digraph_times = [time for times in digraphs.values() for time in times]
                if all_digraph_times:
                    features['digraph_consistency'] = 1.0 / (1.0 + np.std(all_digraph_times))
                else:
                    features['digraph_consistency'] = 0.0
            else:
                features['digraph_consistency'] = 0.0
                
        except Exception as e:
            logger.warning(f"Consistency features extraction error: {e}")
            features['hold_time_cv'] = 0.0
            features['flight_time_cv'] = 0.0
            features['digraph_consistency'] = 0.0
        
        return features
    
    def _extract_pressure_features(self, keystroke_data: List[Dict]) -> Dict:
        """Extract pressure-related features"""
        features = {}
        
        try:
            pressures = []
            for event in keystroke_data:
                pressure = event.get('pressure') or event.get('force')
                if pressure is not None and pressure > 0:
                    pressures.append(pressure)
            
            if pressures and len(pressures) > 1:
                features['pressure_consistency'] = 1.0 / (1.0 + np.std(pressures))
            else:
                features['pressure_consistency'] = 0.8  # Default neutral value
                
        except Exception as e:
            logger.warning(f"Pressure features extraction error: {e}")
            features['pressure_consistency'] = 0.8
        
        return features
    
    def _extract_movement_features(self, mouse_data: List[Dict]) -> Dict:
        """Extract mouse movement features"""
        features = {}
        
        try:
            positions = []
            timestamps = []
            velocities = []
            accelerations = []
            
            for event in mouse_data:
                x = event.get('x', 0)
                y = event.get('y', 0)
                timestamp = event.get('timestamp', 0)
                velocity = event.get('velocity')
                
                positions.append((x, y))
                timestamps.append(timestamp)
                
                if velocity is not None:
                    velocities.append(velocity)
            
            # Calculate velocities if not provided
            if len(velocities) == 0 and len(positions) > 1:
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    dt = timestamps[i] - timestamps[i-1] if len(timestamps) > i else 1
                    
                    if dt > 0:
                        velocity = math.sqrt(dx*dx + dy*dy) / dt
                        velocities.append(velocity)
                        
                        if i > 1 and len(velocities) > 1:
                            dv = velocity - velocities[-2]
                            acceleration = dv / dt
                            accelerations.append(acceleration)
            
            # Velocity features
            if velocities:
                features.update(self._extract_timing_stats('velocity', velocities))
                features['velocity_smoothness'] = self._calculate_smoothness(velocities)
            else:
                features['velocity_mean'] = 0.0
                features['velocity_std'] = 0.0
                features['velocity_median'] = 0.0
                features['velocity_smoothness'] = 0.0
            
            # Acceleration features
            if accelerations:
                features['acceleration_mean'] = float(np.mean(accelerations))
                features['acceleration_std'] = float(np.std(accelerations))
            else:
                features['acceleration_mean'] = 0.0
                features['acceleration_std'] = 0.0
            
            # Movement efficiency
            if len(positions) > 1:
                total_distance = sum(
                    math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                             (positions[i][1] - positions[i-1][1])**2) 
                    for i in range(1, len(positions))
                )
                
                direct_distance = math.sqrt(
                    (positions[-1][0] - positions[0][0])**2 + 
                    (positions[-1][1] - positions[0][1])**2
                )
                
                if total_distance > 0:
                    features['movement_efficiency'] = direct_distance / total_distance
                else:
                    features['movement_efficiency'] = 1.0
            else:
                features['movement_efficiency'] = 1.0
                
        except Exception as e:
            logger.warning(f"Movement features extraction error: {e}")
            # Set default values for all movement features
            for feature in ['velocity_mean', 'velocity_std', 'velocity_median', 
                           'acceleration_mean', 'acceleration_std', 'movement_efficiency', 
                           'velocity_smoothness']:
                if feature not in features:
                    features[feature] = 0.0
        
        return features
    
    def _extract_click_features(self, mouse_data: List[Dict]) -> Dict:
        """Extract mouse click features"""
        features = {}
        
        try:
            clicks = [event for event in mouse_data if event.get('type') == 'click' or event.get('button') is not None]
            
            if clicks:
                # Click timing
                click_durations = []
                for event in clicks:
                    duration = event.get('duration') or event.get('clickDuration')
                    if duration is not None and duration > 0:
                        click_durations.append(duration)
                
                if click_durations:
                    features['click_duration_mean'] = float(np.mean(click_durations))
                    features['click_duration_std'] = float(np.std(click_durations)) if len(click_durations) > 1 else 0.0
                else:
                    features['click_duration_mean'] = 100.0  # Default
                    features['click_duration_std'] = 20.0
                
                # Click types
                left_clicks = sum(1 for click in clicks if click.get('button') == 0)
                right_clicks = sum(1 for click in clicks if click.get('button') == 2)
                total_clicks = len(clicks)
                
                if total_clicks > 0:
                    features['left_click_ratio'] = left_clicks / total_clicks
                    features['right_click_ratio'] = right_clicks / total_clicks
                else:
                    features['left_click_ratio'] = 0.8
                    features['right_click_ratio'] = 0.2
                
                # Inter-click intervals
                if len(clicks) > 1:
                    click_timestamps = [click.get('timestamp', 0) for click in clicks]
                    inter_click_intervals = []
                    for i in range(1, len(click_timestamps)):
                        interval = click_timestamps[i] - click_timestamps[i-1]
                        if interval > 0:
                            inter_click_intervals.append(interval)
                    
                    if inter_click_intervals:
                        features['inter_click_mean'] = float(np.mean(inter_click_intervals))
                        features['inter_click_std'] = float(np.std(inter_click_intervals)) if len(inter_click_intervals) > 1 else 0.0
                    else:
                        features['inter_click_mean'] = 1000.0
                        features['inter_click_std'] = 200.0
                else:
                    features['inter_click_mean'] = 1000.0
                    features['inter_click_std'] = 200.0
            else:
                # No clicks found - set defaults
                features['click_duration_mean'] = 100.0
                features['click_duration_std'] = 20.0
                features['left_click_ratio'] = 0.8
                features['right_click_ratio'] = 0.2
                features['inter_click_mean'] = 1000.0
                features['inter_click_std'] = 200.0
                
        except Exception as e:
            logger.warning(f"Click features extraction error: {e}")
            features['click_duration_mean'] = 100.0
            features['click_duration_std'] = 20.0
            features['left_click_ratio'] = 0.8
            features['right_click_ratio'] = 0.2
            features['inter_click_mean'] = 1000.0
            features['inter_click_std'] = 200.0
        
        return features
    
    def _extract_trajectory_features(self, mouse_data: List[Dict]) -> Dict:
        """Extract mouse trajectory features"""
        features = {}
        
        try:
            positions = [(event.get('x', 0), event.get('y', 0)) for event in mouse_data]
            
            if len(positions) < 3:
                features['curvature_mean'] = 0.0
                features['curvature_std'] = 0.0
                features['avg_direction_change'] = 0.0
                features['direction_change_variance'] = 0.0
                return features
            
            # Curvature analysis
            curvatures = []
            for i in range(1, len(positions) - 1):
                p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
                curvature = self._calculate_curvature(p1, p2, p3)
                curvatures.append(curvature)
            
            if curvatures:
                features['curvature_mean'] = float(np.mean(curvatures))
                features['curvature_std'] = float(np.std(curvatures)) if len(curvatures) > 1 else 0.0
            else:
                features['curvature_mean'] = 0.0
                features['curvature_std'] = 0.0
            
            # Direction changes
            directions = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                
                if dx != 0 or dy != 0:
                    angle = math.atan2(dy, dx)
                    directions.append(angle)
            
            if len(directions) > 1:
                direction_changes = []
                for i in range(1, len(directions)):
                    change = abs(directions[i] - directions[i-1])
                    change = min(change, 2*math.pi - change)  # Normalize to [0, pi]
                    direction_changes.append(change)
                
                if direction_changes:
                    features['avg_direction_change'] = float(np.mean(direction_changes))
                    features['direction_change_variance'] = float(np.var(direction_changes))
                else:
                    features['avg_direction_change'] = 0.0
                    features['direction_change_variance'] = 0.0
            else:
                features['avg_direction_change'] = 0.0
                features['direction_change_variance'] = 0.0
                
        except Exception as e:
            logger.warning(f"Trajectory features extraction error: {e}")
            features['curvature_mean'] = 0.0
            features['curvature_std'] = 0.0
            features['avg_direction_change'] = 0.0
            features['direction_change_variance'] = 0.0
        
        return features
    
    def _extract_behavioral_patterns(self, mouse_data: List[Dict]) -> Dict:
        """Extract higher-level behavioral patterns"""
        features = {}
        
        try:
            # Dwell time analysis
            hover_events = [event for event in mouse_data if event.get('type') == 'hover']
            if hover_events:
                dwell_times = [event.get('duration', 0) for event in hover_events if event.get('duration', 0) > 0]
                if dwell_times:
                    features['dwell_time_mean'] = float(np.mean(dwell_times))
                else:
                    features['dwell_time_mean'] = 500.0
            else:
                features['dwell_time_mean'] = 500.0
            
            # Movement area and centrality
            positions = [(event.get('x', 0), event.get('y', 0)) for event in mouse_data]
            if len(positions) > 10:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                features['movement_area'] = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                
                center_x, center_y = np.mean(x_coords), np.mean(y_coords)
                distances_from_center = [
                    math.sqrt((x - center_x)**2 + (y - center_y)**2) 
                    for x, y in positions
                ]
                features['movement_centrality'] = float(np.mean(distances_from_center))
            else:
                features['movement_area'] = 10000.0
                features['movement_centrality'] = 300.0
                
        except Exception as e:
            logger.warning(f"Behavioral patterns extraction error: {e}")
            features['dwell_time_mean'] = 500.0
            features['movement_area'] = 10000.0
            features['movement_centrality'] = 300.0
        
        return features
    
    def _normalize_keystroke_features(self, features: Dict) -> Dict:
        """Ensure keystroke features match expected dimensions exactly"""
        normalized = {}
        
        for feature_name in self.KEYSTROKE_FEATURES:
            if feature_name in features:
                value = features[feature_name]
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    normalized[feature_name] = float(value)
                else:
                    normalized[feature_name] = self._get_default_keystroke_value(feature_name)
            else:
                normalized[feature_name] = self._get_default_keystroke_value(feature_name)
        
        return normalized
    
    def _normalize_mouse_features(self, features: Dict) -> Dict:
        """Ensure mouse features match expected dimensions exactly"""
        normalized = {}
        
        for feature_name in self.MOUSE_FEATURES:
            if feature_name in features:
                value = features[feature_name]
                if isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value)):
                    normalized[feature_name] = float(value)
                else:
                    normalized[feature_name] = self._get_default_mouse_value(feature_name)
            else:
                normalized[feature_name] = self._get_default_mouse_value(feature_name)
        
        return normalized
    
    def _get_default_keystroke_value(self, feature_name: str) -> float:
        """Get default value for missing keystroke features"""
        defaults = {
            'hold_time_mean': 100.0, 'hold_time_std': 20.0, 'hold_time_median': 95.0,
            'flight_time_mean': 80.0, 'flight_time_std': 25.0, 'flight_time_median': 75.0,
            'typing_speed_wpm': 40.0, 'typing_speed_cpm': 200.0,
            'rhythm_consistency': 0.8, 'burst_ratio': 0.3, 'pause_ratio': 0.2,
            'avg_pause_duration': 200.0, 'speed_variance': 5.0, 'speed_trend': 0.0,
            'digraph_consistency': 0.75, 'hold_time_cv': 0.2, 'flight_time_cv': 0.3,
            'pressure_consistency': 0.8
        }
        return defaults.get(feature_name, 0.0)
    
    def _get_default_mouse_value(self, feature_name: str) -> float:
        """Get default value for missing mouse features"""
        defaults = {
            'velocity_mean': 2.5, 'velocity_std': 1.0, 'velocity_median': 2.0,
            'acceleration_mean': 1.0, 'acceleration_std': 0.5,
            'movement_efficiency': 0.8, 'curvature_mean': 0.3, 'curvature_std': 0.2,
            'avg_direction_change': 0.5, 'direction_change_variance': 0.3,
            'click_duration_mean': 100.0, 'click_duration_std': 30.0,
            'left_click_ratio': 0.8, 'right_click_ratio': 0.2,
            'inter_click_mean': 1000.0, 'inter_click_std': 200.0,
            'dwell_time_mean': 500.0, 'movement_area': 10000.0,
            'movement_centrality': 300.0, 'velocity_smoothness': 0.8
        }
        return defaults.get(feature_name, 0.0)
    
    def _get_empty_keystroke_features(self) -> Dict:
        """Return empty keystroke features with default values"""
        return {name: self._get_default_keystroke_value(name) for name in self.KEYSTROKE_FEATURES}
    
    def _get_empty_mouse_features(self) -> Dict:
        """Return empty mouse features with default values"""
        return {name: self._get_default_mouse_value(name) for name in self.MOUSE_FEATURES}
    
    def get_combined_features(self, keystroke_features: Dict, mouse_features: Dict) -> Dict:
        """Combine keystroke and mouse features into a single feature dict"""
        combined = {}
        
        # Normalize individual feature sets
        ks_features = self._normalize_keystroke_features(keystroke_features)
        mouse_features = self._normalize_mouse_features(mouse_features)
        
        # Combine in the correct order
        combined.update(ks_features)
        combined.update(mouse_features)
        
        assert len(combined) == self.FEATURE_COUNT, f"Combined features count mismatch: {len(combined)} != {self.FEATURE_COUNT}"
        
        return combined
    
    def get_feature_vector(self, keystroke_features: Dict, mouse_features: Dict) -> np.ndarray:
        """Convert features to consistent numpy array"""
        combined = self.get_combined_features(keystroke_features, mouse_features)
        
        # Create feature vector in consistent order
        feature_vector = []
        for feature_name in self.ALL_FEATURES:
            feature_vector.append(combined[feature_name])
        
        return np.array(feature_vector, dtype=np.float32)
    
    def validate_feature_dimensions(self, features: Dict) -> bool:
        """Validate that features have correct dimensions"""
        if isinstance(features, dict):
            return len(features) == self.FEATURE_COUNT
        return False
    
    def fix_feature_dimensions(self, features: Dict) -> Dict:
        """Fix feature dimensions to match expected count"""
        if len(features) == self.FEATURE_COUNT:
            return features
        
        # Create properly sized feature dict
        fixed_features = {}
        
        # Add keystroke features
        for feature_name in self.KEYSTROKE_FEATURES:
            if feature_name in features:
                fixed_features[feature_name] = features[feature_name]
            else:
                fixed_features[feature_name] = self._get_default_keystroke_value(feature_name)
        
        # Add mouse features
        for feature_name in self.MOUSE_FEATURES:
            if feature_name in features:
                fixed_features[feature_name] = features[feature_name]
            else:
                fixed_features[feature_name] = self._get_default_mouse_value(feature_name)
        
        return fixed_features
    
    # Helper methods for statistical calculations
    @staticmethod
    def _calculate_trend(values: List[float]) -> float:
        """Calculate trend in values using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_smoothness(values: List[float]) -> float:
        """Calculate smoothness of a value sequence"""
        if len(values) < 3:
            return 1.0
        
        try:
            jerk = np.diff(values, n=2)  # Second derivative
            return 1.0 / (1.0 + np.mean(np.abs(jerk)))
        except:
            return 1.0
    
    @staticmethod
    def _calculate_curvature(p1: Tuple[float, float], p2: Tuple[float, float], 
                           p3: Tuple[float, float]) -> float:
        """Calculate curvature at point p2 given three consecutive points"""
        try:
            # Vector from p1 to p2
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Vector from p2 to p3
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Cross product magnitude
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            # Magnitudes
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 == 0:
                return 0.0
            
            return cross / (mag1 * mag2)
        except:
            return 0.0
    
    def get_feature_info(self) -> Dict:
        """Get information about the feature extractor configuration"""
        return {
            'total_features': self.FEATURE_COUNT,
            'keystroke_features': len(self.KEYSTROKE_FEATURES),
            'mouse_features': len(self.MOUSE_FEATURES),
            'keystroke_feature_names': self.KEYSTROKE_FEATURES,
            'mouse_feature_names': self.MOUSE_FEATURES,
            'all_feature_names': self.ALL_FEATURES
        }
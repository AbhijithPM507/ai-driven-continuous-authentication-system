import numpy as np
from scipy import stats
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import math
from datetime import datetime, timedelta

class BehavioralDriftDetector:
    """Detects drift in user behavioral patterns using statistical tests"""
    
    def __init__(self, window_size: int = 100, alpha: float = 0.05, 
                 min_samples: int = 30):
        self.window_size = window_size
        self.alpha = alpha  # Statistical significance level
        self.min_samples = min_samples
        
        # Sliding windows for different feature types
        self.keystroke_window = deque(maxlen=window_size)
        self.mouse_window = deque(maxlen=window_size)
        
        # Reference distributions (established during calibration)
        self.reference_keystroke = None
        self.reference_mouse = None
        
        # Drift detection state
        self.drift_detected = False
        self.last_drift_time = None
        self.drift_score = 0.0
        
        # Feature importance weights
        self.feature_weights = {
            'keystroke': {
                'hold_time_mean': 1.0,
                'flight_time_mean': 1.0,
                'typing_speed_wpm': 0.8,
                'rhythm_consistency': 0.9,
                'digraph_timing': 0.7
            },
            'mouse': {
                'velocity_mean': 1.0,
                'acceleration_mean': 0.8,
                'movement_efficiency': 0.9,
                'curvature_mean': 0.7,
                'click_duration_mean': 0.8
            }
        }
    
    def set_reference_baseline(self, keystroke_features: List[Dict], 
                              mouse_features: List[Dict]):
        """Set reference distributions from calibration data"""
        if keystroke_features:
            self.reference_keystroke = self._calculate_feature_statistics(keystroke_features)
        
        if mouse_features:
            self.reference_mouse = self._calculate_feature_statistics(mouse_features)
    
    def add_sample(self, features: Dict, data_type: str):
        """Add new behavioral sample for drift monitoring"""
        if data_type == 'keystroke':
            self.keystroke_window.append(features)
        elif data_type == 'mouse':
            self.mouse_window.append(features)
        
        # Check for drift if we have enough samples
        if self._has_sufficient_samples():
            self._check_for_drift()
    
    def _calculate_feature_statistics(self, features_list: List[Dict]) -> Dict:
        """Calculate statistical properties of features"""
        if not features_list:
            return {}
        
        stats_dict = {}
        
        # Get all unique feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
        
        for feature_name in all_features:
            values = [f.get(feature_name, 0) for f in features_list]
            values = [v for v in values if v is not None and not math.isnan(v)]
            
            if len(values) > 1:
                stats_dict[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values),
                    'samples': len(values)
                }
        
        return stats_dict
    
    def _has_sufficient_samples(self) -> bool:
        """Check if we have enough samples for drift detection"""
        return (len(self.keystroke_window) >= self.min_samples or 
                len(self.mouse_window) >= self.min_samples)
    
    def _check_for_drift(self):
        """Main drift detection logic"""
        drift_scores = []
        
        # Check keystroke drift
        if (self.reference_keystroke and 
            len(self.keystroke_window) >= self.min_samples):
            keystroke_drift = self._detect_distribution_drift(
                list(self.keystroke_window), 
                self.reference_keystroke, 
                'keystroke'
            )
            drift_scores.append(keystroke_drift)
        
        # Check mouse drift
        if (self.reference_mouse and 
            len(self.mouse_window) >= self.min_samples):
            mouse_drift = self._detect_distribution_drift(
                list(self.mouse_window), 
                self.reference_mouse, 
                'mouse'
            )
            drift_scores.append(mouse_drift)
        
        # Calculate overall drift score
        if drift_scores:
            self.drift_score = np.mean(drift_scores)
            
            # Determine if drift is significant
            drift_threshold = 0.3  # Configurable threshold
            if self.drift_score > drift_threshold:
                self.drift_detected = True
                self.last_drift_time = datetime.now()
        
    def _detect_distribution_drift(self, current_samples: List[Dict], 
                                  reference_stats: Dict, data_type: str) -> float:
        """Detect drift between current samples and reference distribution"""
        current_stats = self._calculate_feature_statistics(current_samples)
        
        if not current_stats or not reference_stats:
            return 0.0
        
        drift_indicators = []
        weights = self.feature_weights.get(data_type, {})
        
        for feature_name in reference_stats.keys():
            if feature_name in current_stats:
                feature_drift = self._calculate_feature_drift(
                    current_stats[feature_name],
                    reference_stats[feature_name],
                    feature_name
                )
                
                # Apply feature weight
                weight = weights.get(feature_name, 0.5)
                weighted_drift = feature_drift * weight
                drift_indicators.append(weighted_drift)
        
        return np.mean(drift_indicators) if drift_indicators else 0.0
    
    def _calculate_feature_drift(self, current_stats: Dict, 
                               reference_stats: Dict, feature_name: str) -> float:
        """Calculate drift score for a specific feature"""
        drift_score = 0.0
        
        # Mean shift detection
        mean_drift = self._detect_mean_shift(current_stats, reference_stats)
        drift_score += mean_drift * 0.4
        
        # Variance change detection
        variance_drift = self._detect_variance_change(current_stats, reference_stats)
        drift_score += variance_drift * 0.3
        
        # Distribution shape change
        shape_drift = self._detect_shape_change(current_stats, reference_stats)
        drift_score += shape_drift * 0.3
        
        return min(drift_score, 1.0)  # Cap at 1.0
    
    def _detect_mean_shift(self, current_stats: Dict, reference_stats: Dict) -> float:
        """Detect shift in mean using Cohen's d effect size"""
        current_mean = current_stats['mean']
        reference_mean = reference_stats['mean']
        
        # Pooled standard deviation
        current_std = current_stats['std']
        reference_std = reference_stats['std']
        
        if current_std == 0 and reference_std == 0:
            return 0.0
        
        pooled_std = math.sqrt((current_std**2 + reference_std**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d effect size
        cohens_d = abs(current_mean - reference_mean) / pooled_std
        
        # Convert to drift score (0-1)
        # Small effect: d=0.2, Medium: d=0.5, Large: d=0.8
        drift_score = min(cohens_d / 0.8, 1.0)
        
        return drift_score
    
    def _detect_variance_change(self, current_stats: Dict, reference_stats: Dict) -> float:
        """Detect change in variance using F-test approach"""
        current_var = current_stats['std']**2
        reference_var = reference_stats['std']**2
        
        if current_var == 0 and reference_var == 0:
            return 0.0
        
        if reference_var == 0:
            return 1.0 if current_var > 0 else 0.0
        
        # Variance ratio
        var_ratio = current_var / reference_var
        
        # Log ratio for symmetric treatment
        log_ratio = abs(math.log(var_ratio))
        
        # Convert to drift score
        # log(2) ≈ 0.69 represents doubling/halving of variance
        drift_score = min(log_ratio / 0.69, 1.0)
        
        return drift_score
    
    def _detect_shape_change(self, current_stats: Dict, reference_stats: Dict) -> float:
        """Detect change in distribution shape using skewness and kurtosis"""
        skew_diff = abs(current_stats['skewness'] - reference_stats['skewness'])
        kurt_diff = abs(current_stats['kurtosis'] - reference_stats['kurtosis'])
        
        # Normalize skewness difference (typical range -2 to 2)
        skew_score = min(skew_diff / 2.0, 1.0)
        
        # Normalize kurtosis difference (typical range -2 to 2)
        kurt_score = min(kurt_diff / 2.0, 1.0)
        
        # Combined shape drift score
        shape_drift = (skew_score + kurt_score) / 2
        
        return shape_drift
    
    def perform_statistical_tests(self, current_samples: List[Dict], 
                                 reference_samples: List[Dict], 
                                 feature_name: str) -> Dict:
        """Perform formal statistical tests for drift detection"""
        results = {}
        
        # Extract feature values
        current_values = [s.get(feature_name, 0) for s in current_samples]
        reference_values = [s.get(feature_name, 0) for s in reference_samples]
        
        # Remove invalid values
        current_values = [v for v in current_values if not math.isnan(v)]
        reference_values = [v for v in reference_values if not math.isnan(v)]
        
        if len(current_values) < 5 or len(reference_values) < 5:
            return results
        
        try:
            # Kolmogorov-Smirnov test (distribution comparison)
            ks_stat, ks_pvalue = stats.ks_2samp(current_values, reference_values)
            results['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'significant': ks_pvalue < self.alpha
            }
            
            # Mann-Whitney U test (median comparison)
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                current_values, reference_values, alternative='two-sided'
            )
            results['mannwhitney_test'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_pvalue),
                'significant': mw_pvalue < self.alpha
            }
            
            # Levene's test (variance comparison)
            levene_stat, levene_pvalue = stats.levene(current_values, reference_values)
            results['levene_test'] = {
                'statistic': float(levene_stat),
                'p_value': float(levene_pvalue),
                'significant': levene_pvalue < self.alpha
            }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def get_drift_analysis(self) -> Dict:
        """Get comprehensive drift analysis"""
        analysis = {
            'drift_detected': self.drift_detected,
            'drift_score': float(self.drift_score),
            'last_drift_time': self.last_drift_time.isoformat() if self.last_drift_time else None,
            'sample_counts': {
                'keystroke': len(self.keystroke_window),
                'mouse': len(self.mouse_window)
            }
        }
        
        # Add detailed feature analysis if drift detected
        if self.drift_detected:
            analysis['feature_analysis'] = self._get_detailed_feature_analysis()
        
        return analysis
    
    def _get_detailed_feature_analysis(self) -> Dict:
        """Get detailed analysis of which features are drifting"""
        feature_analysis = {}
        
        # Analyze keystroke features
        if (self.reference_keystroke and len(self.keystroke_window) >= self.min_samples):
            current_keystroke = self._calculate_feature_statistics(list(self.keystroke_window))
            keystroke_drift = {}
            
            for feature_name in self.reference_keystroke.keys():
                if feature_name in current_keystroke:
                    drift_score = self._calculate_feature_drift(
                        current_keystroke[feature_name],
                        self.reference_keystroke[feature_name],
                        feature_name
                    )
                    keystroke_drift[feature_name] = {
                        'drift_score': float(drift_score),
                        'current_mean': float(current_keystroke[feature_name]['mean']),
                        'reference_mean': float(self.reference_keystroke[feature_name]['mean']),
                        'mean_change_percent': float(
                            abs(current_keystroke[feature_name]['mean'] - 
                                self.reference_keystroke[feature_name]['mean']) /
                            max(self.reference_keystroke[feature_name]['mean'], 1e-6) * 100
                        )
                    }
            
            feature_analysis['keystroke'] = keystroke_drift
        
        # Analyze mouse features
        if (self.reference_mouse and len(self.mouse_window) >= self.min_samples):
            current_mouse = self._calculate_feature_statistics(list(self.mouse_window))
            mouse_drift = {}
            
            for feature_name in self.reference_mouse.keys():
                if feature_name in current_mouse:
                    drift_score = self._calculate_feature_drift(
                        current_mouse[feature_name],
                        self.reference_mouse[feature_name],
                        feature_name
                    )
                    mouse_drift[feature_name] = {
                        'drift_score': float(drift_score),
                        'current_mean': float(current_mouse[feature_name]['mean']),
                        'reference_mean': float(self.reference_mouse[feature_name]['mean']),
                        'mean_change_percent': float(
                            abs(current_mouse[feature_name]['mean'] - 
                                self.reference_mouse[feature_name]['mean']) /
                            max(self.reference_mouse[feature_name]['mean'], 1e-6) * 100
                        )
                    }
            
            feature_analysis['mouse'] = mouse_drift
        
        return feature_analysis
    
    def reset_drift_detection(self):
        """Reset drift detection state (called after retraining)"""
        self.drift_detected = False
        self.drift_score = 0.0
        self.last_drift_time = None
        
        # Update reference distributions with recent data
        if len(self.keystroke_window) >= self.min_samples:
            self.reference_keystroke = self._calculate_feature_statistics(
                list(self.keystroke_window)
            )
        
        if len(self.mouse_window) >= self.min_samples:
            self.reference_mouse = self._calculate_feature_statistics(
                list(self.mouse_window)
            )
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained based on drift analysis"""
        if not self.drift_detected:
            return False
        
        # Check if enough time has passed since last drift detection
        if self.last_drift_time:
            time_since_drift = datetime.now() - self.last_drift_time
            if time_since_drift < timedelta(minutes=30):  # Minimum time between retraining
                return False
        
        # Check drift severity
        severe_drift_threshold = 0.5
        if self.drift_score > severe_drift_threshold:
            return True
        
        # Check if drift is consistent across features
        if self.drift_score > 0.3:
            analysis = self._get_detailed_feature_analysis()
            
            # Count features with significant drift
            significant_drift_count = 0
            total_features = 0
            
            for data_type in ['keystroke', 'mouse']:
                if data_type in analysis:
                    for feature, info in analysis[data_type].items():
                        total_features += 1
                        if info['drift_score'] > 0.4:
                            significant_drift_count += 1
            
            # Retrain if more than 50% of features show significant drift
            if total_features > 0 and significant_drift_count / total_features > 0.5:
                return True
        
        return False
    
    def get_adaptation_suggestions(self) -> List[str]:
        """Get suggestions for adapting to detected drift"""
        suggestions = []
        
        if not self.drift_detected:
            return suggestions
        
        analysis = self._get_detailed_feature_analysis()
        
        # Analyze keystroke drift patterns
        if 'keystroke' in analysis:
            keystroke_features = analysis['keystroke']
            
            # Check for typing speed changes
            if 'typing_speed_wpm' in keystroke_features:
                speed_change = keystroke_features['typing_speed_wpm']['mean_change_percent']
                if speed_change > 20:
                    suggestions.append("Significant typing speed change detected - consider stress or fatigue factors")
            
            # Check for timing pattern changes
            timing_features = ['hold_time_mean', 'flight_time_mean', 'rhythm_consistency']
            timing_drift = [keystroke_features.get(f, {}).get('drift_score', 0) 
                           for f in timing_features]
            
            if np.mean(timing_drift) > 0.4:
                suggestions.append("Typing rhythm patterns have changed - possible injury or adaptation")
        
        # Analyze mouse drift patterns
        if 'mouse' in analysis:
            mouse_features = analysis['mouse']
            
            # Check for movement efficiency changes
            if 'movement_efficiency' in mouse_features:
                efficiency_change = mouse_features['movement_efficiency']['mean_change_percent']
                if efficiency_change > 25:
                    suggestions.append("Mouse movement efficiency changed - possible device or workspace change")
            
            # Check for velocity/acceleration changes
            motion_features = ['velocity_mean', 'acceleration_mean']
            motion_drift = [mouse_features.get(f, {}).get('drift_score', 0) 
                           for f in motion_features]
            
            if np.mean(motion_drift) > 0.4:
                suggestions.append("Mouse movement dynamics changed - consider device sensitivity settings")
        
        # General suggestions
        if self.drift_score > 0.6:
            suggestions.append("Significant behavioral changes detected - recommend model retraining")
        elif self.drift_score > 0.4:
            suggestions.append("Moderate behavioral drift - monitor closely and consider gradual adaptation")
        
        return suggestions
"""
Trend Analysis - Objective trend detection in observed data.

This module detects temporal and sequential trends without making
judgments about whether trends are positive or negative.
"""

from typing import Dict, List, Any, Tuple
import math


class TrendAnalyzer:
    """Detects objective trends in sequential data."""
    
    def __init__(self):
        self.trend_cache = {}
    
    def detect_temporal_trends(self, time_series: List[Tuple[float, Any]]) -> Dict[str, Any]:
        """
        Detect trends over time in sequential data.
        
        Args:
            time_series: List of (timestamp, value) tuples
            
        Returns:
            Dictionary with temporal trend analysis
        """
        if len(time_series) < 3:
            return {'insufficient_data': True}
        
        # Sort by timestamp
        sorted_series = sorted(time_series, key=lambda x: x[0])
        
        timestamps = [point[0] for point in sorted_series]
        values = [point[1] for point in sorted_series]
        
        # Detect different types of trends
        linear_trend = self._detect_linear_trend(timestamps, values)
        monotonic_trend = self._detect_monotonic_trend(values)
        cyclical_patterns = self._detect_cyclical_patterns(values)
        change_points = self._detect_change_points(values)
        
        return {
            'insufficient_data': False,
            'data_points': len(sorted_series),
            'time_span': timestamps[-1] - timestamps[0],
            'linear_trend': linear_trend,
            'monotonic_trend': monotonic_trend,
            'cyclical_patterns': cyclical_patterns,
            'change_points': change_points,
            'trend_consistency': self._calculate_trend_consistency(values)
        }
    
    def detect_sequence_trends(self, sequence: List[Any]) -> Dict[str, Any]:
        """
        Detect trends in sequential data (non-temporal).
        
        Args:
            sequence: List of sequential values
            
        Returns:
            Dictionary with sequence trend analysis
        """
        if len(sequence) < 3:
            return {'insufficient_data': True}
        
        # Convert to numerical if possible for trend analysis
        numerical_sequence = self._extract_numerical_features(sequence)
        
        if not numerical_sequence:
            return {'non_numerical_data': True}
        
        # Analyze trends in the numerical representation
        directional_trend = self._analyze_directional_trend(numerical_sequence)
        variability_trend = self._analyze_variability_trend(numerical_sequence)
        pattern_repetition = self._detect_pattern_repetition(sequence)
        
        return {
            'insufficient_data': False,
            'non_numerical_data': False,
            'sequence_length': len(sequence),
            'directional_trend': directional_trend,
            'variability_trend': variability_trend,
            'pattern_repetition': pattern_repetition,
            'trend_strength': self._calculate_trend_strength(numerical_sequence)
        }
    
    def detect_performance_trends(self, performance_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Detect trends in performance metrics over time.
        
        Args:
            performance_metrics: List of metric dictionaries
            
        Returns:
            Dictionary with performance trend analysis
        """
        if len(performance_metrics) < 3:
            return {'insufficient_data': True}
        
        # Extract individual metric trends
        metric_trends = {}
        
        # Get all metric names
        all_metrics = set()
        for metrics_dict in performance_metrics:
            all_metrics.update(metrics_dict.keys())
        
        # Analyze trend for each metric
        for metric_name in all_metrics:
            metric_values = []
            for i, metrics_dict in enumerate(performance_metrics):
                if metric_name in metrics_dict:
                    metric_values.append((i, metrics_dict[metric_name]))
            
            if len(metric_values) >= 3:
                timestamps = [point[0] for point in metric_values]
                values = [point[1] for point in metric_values]
                
                metric_trends[metric_name] = {
                    'linear_trend': self._detect_linear_trend(timestamps, values),
                    'trend_direction': self._determine_trend_direction(values),
                    'volatility': self._calculate_volatility(values),
                    'data_points': len(metric_values)
                }
        
        # Overall performance trend summary
        overall_trend = self._summarize_overall_trend(metric_trends)
        
        return {
            'insufficient_data': False,
            'metrics_analyzed': len(metric_trends),
            'individual_metric_trends': metric_trends,
            'overall_trend_summary': overall_trend
        }
    
    # Private helper methods
    
    def _detect_linear_trend(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """Detect linear trend using simple linear regression."""
        n = len(x_values)
        if n < 2:
            return {'no_trend': True}
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate slope and intercept
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return {'no_trend': True}
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [slope * x + intercept for x in x_values]
        ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'no_trend': False,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'trend_strength': abs(slope),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        }
    
    def _detect_monotonic_trend(self, values: List[float]) -> Dict[str, Any]:
        """Detect monotonic trends (consistently increasing or decreasing)."""
        if len(values) < 2:
            return {'insufficient_data': True}
        
        increasing_count = 0
        decreasing_count = 0
        equal_count = 0
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increasing_count += 1
            elif values[i] < values[i-1]:
                decreasing_count += 1
            else:
                equal_count += 1
        
        total_comparisons = len(values) - 1
        
        # Determine monotonic trend type
        if increasing_count == total_comparisons:
            trend_type = 'strictly_increasing'
        elif decreasing_count == total_comparisons:
            trend_type = 'strictly_decreasing'
        elif increasing_count >= total_comparisons * 0.8:
            trend_type = 'mostly_increasing'
        elif decreasing_count >= total_comparisons * 0.8:
            trend_type = 'mostly_decreasing'
        else:
            trend_type = 'mixed'
        
        return {
            'insufficient_data': False,
            'trend_type': trend_type,
            'increasing_steps': increasing_count,
            'decreasing_steps': decreasing_count,
            'equal_steps': equal_count,
            'monotonic_ratio': max(increasing_count, decreasing_count) / total_comparisons
        }
    
    def _detect_cyclical_patterns(self, values: List[float]) -> Dict[str, Any]:
        """Detect cyclical patterns in the data."""
        if len(values) < 6:  # Need at least 6 points to detect cycles
            return {'insufficient_data': True}
        
        # Simple cycle detection using autocorrelation
        max_lag = min(len(values) // 2, 10)  # Check up to half the data length or 10
        
        autocorrelations = []
        for lag in range(1, max_lag + 1):
            autocorr = self._calculate_autocorrelation(values, lag)
            autocorrelations.append((lag, autocorr))
        
        # Find peaks in autocorrelation (potential cycle lengths)
        potential_cycles = []
        for i, (lag, autocorr) in enumerate(autocorrelations):
            if abs(autocorr) > 0.3:  # Threshold for significant correlation
                potential_cycles.append({'lag': lag, 'correlation': autocorr})
        
        return {
            'insufficient_data': False,
            'potential_cycles': potential_cycles,
            'max_autocorrelation': max(abs(ac[1]) for ac in autocorrelations),
            'cyclical_evidence': len(potential_cycles) > 0
        }
    
    def _detect_change_points(self, values: List[float]) -> Dict[str, Any]:
        """Detect significant change points in the data."""
        if len(values) < 4:
            return {'insufficient_data': True}
        
        change_points = []
        
        # Simple change point detection using moving averages
        window_size = max(2, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            # Calculate means before and after potential change point
            before_mean = sum(values[i-window_size:i]) / window_size
            after_mean = sum(values[i:i+window_size]) / window_size
            
            # Calculate change magnitude
            change_magnitude = abs(after_mean - before_mean)
            
            # Use standard deviation as threshold
            overall_std = math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values))
            
            if change_magnitude > overall_std:  # Significant change
                change_points.append({
                    'position': i,
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'change_magnitude': change_magnitude
                })
        
        return {
            'insufficient_data': False,
            'change_points_detected': len(change_points),
            'change_points': change_points,
            'has_significant_changes': len(change_points) > 0
        }
    
    def _calculate_trend_consistency(self, values: List[float]) -> float:
        """Calculate how consistent the trend is."""
        if len(values) < 3:
            return 0
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                directions.append(1)
            elif values[i] < values[i-1]:
                directions.append(-1)
            else:
                directions.append(0)
        
        if not directions:
            return 0
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1] and directions[i] != 0 and directions[i-1] != 0:
                direction_changes += 1
        
        # Consistency is inverse of direction changes
        max_possible_changes = len(directions) - 1
        consistency = 1 - (direction_changes / max_possible_changes) if max_possible_changes > 0 else 1
        
        return consistency
    
    def _extract_numerical_features(self, sequence: List[Any]) -> List[float]:
        """Extract numerical features from a sequence for trend analysis."""
        numerical_features = []
        
        for item in sequence:
            if isinstance(item, (int, float)):
                numerical_features.append(float(item))
            elif isinstance(item, str):
                # Use string length as a numerical feature
                numerical_features.append(float(len(item)))
            elif hasattr(item, '__len__'):
                # Use length for sequences
                numerical_features.append(float(len(item)))
            else:
                # Use hash as a numerical representation (not ideal but objective)
                numerical_features.append(float(hash(str(item)) % 1000))
        
        return numerical_features
    
    def _analyze_directional_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze the directional trend in numerical values."""
        if len(values) < 2:
            return {'insufficient_data': True}
        
        differences = [values[i] - values[i-1] for i in range(1, len(values))]
        
        positive_diffs = sum(1 for d in differences if d > 0)
        negative_diffs = sum(1 for d in differences if d < 0)
        zero_diffs = sum(1 for d in differences if d == 0)
        
        total_diffs = len(differences)
        
        if positive_diffs > negative_diffs:
            primary_direction = 'increasing'
        elif negative_diffs > positive_diffs:
            primary_direction = 'decreasing'
        else:
            primary_direction = 'stable'
        
        return {
            'insufficient_data': False,
            'primary_direction': primary_direction,
            'positive_changes': positive_diffs,
            'negative_changes': negative_diffs,
            'no_changes': zero_diffs,
            'directional_consistency': max(positive_diffs, negative_diffs) / total_diffs
        }
    
    def _analyze_variability_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trends in variability over time."""
        if len(values) < 4:
            return {'insufficient_data': True}
        
        # Calculate rolling standard deviations
        window_size = max(2, len(values) // 3)
        rolling_stds = []
        
        for i in range(window_size, len(values) + 1):
            window_values = values[i-window_size:i]
            window_mean = sum(window_values) / len(window_values)
            window_std = math.sqrt(sum((v - window_mean)**2 for v in window_values) / len(window_values))
            rolling_stds.append(window_std)
        
        if len(rolling_stds) < 2:
            return {'insufficient_data': True}
        
        # Analyze trend in variability
        variability_trend = self._detect_linear_trend(list(range(len(rolling_stds))), rolling_stds)
        
        return {
            'insufficient_data': False,
            'variability_trend': variability_trend,
            'initial_variability': rolling_stds[0],
            'final_variability': rolling_stds[-1],
            'variability_change': rolling_stds[-1] - rolling_stds[0]
        }
    
    def _detect_pattern_repetition(self, sequence: List[Any]) -> Dict[str, Any]:
        """Detect repeating patterns in the sequence."""
        if len(sequence) < 4:
            return {'insufficient_data': True}
        
        # Look for repeating subsequences
        pattern_counts = {}
        max_pattern_length = min(len(sequence) // 2, 5)  # Limit pattern length
        
        for pattern_length in range(2, max_pattern_length + 1):
            for start in range(len(sequence) - pattern_length + 1):
                pattern = tuple(sequence[start:start + pattern_length])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Find most frequent patterns
        repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count > 1}
        
        return {
            'insufficient_data': False,
            'repeated_patterns_found': len(repeated_patterns),
            'most_frequent_patterns': sorted(repeated_patterns.items(), key=lambda x: x[1], reverse=True)[:5],
            'repetition_ratio': len(repeated_patterns) / len(pattern_counts) if pattern_counts else 0
        }
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate overall trend strength."""
        if len(values) < 3:
            return 0
        
        # Use correlation with time as trend strength measure
        time_points = list(range(len(values)))
        correlation = self._calculate_correlation(time_points, values)
        
        return abs(correlation)
    
    def _determine_trend_direction(self, values: List[float]) -> str:
        """Determine overall trend direction."""
        if len(values) < 2:
            return 'unknown'
        
        first_half_mean = sum(values[:len(values)//2]) / (len(values)//2)
        second_half_mean = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half_mean > first_half_mean * 1.05:  # 5% threshold
            return 'increasing'
        elif second_half_mean < first_half_mean * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation of changes)."""
        if len(values) < 2:
            return 0
        
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        
        if not changes:
            return 0
        
        mean_change = sum(changes) / len(changes)
        variance = sum((c - mean_change)**2 for c in changes) / len(changes)
        
        return math.sqrt(variance)
    
    def _summarize_overall_trend(self, metric_trends: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize overall trend across all metrics."""
        if not metric_trends:
            return {'no_data': True}
        
        # Count trend directions
        trend_directions = {}
        trend_strengths = []
        
        for metric_name, trend_data in metric_trends.items():
            direction = trend_data.get('trend_direction', 'unknown')
            trend_directions[direction] = trend_directions.get(direction, 0) + 1
            
            linear_trend = trend_data.get('linear_trend', {})
            if 'r_squared' in linear_trend:
                trend_strengths.append(linear_trend['r_squared'])
        
        # Determine dominant trend direction
        if trend_directions:
            dominant_direction = max(trend_directions.items(), key=lambda x: x[1])[0]
        else:
            dominant_direction = 'unknown'
        
        # Calculate average trend strength
        avg_trend_strength = sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0
        
        return {
            'no_data': False,
            'dominant_trend_direction': dominant_direction,
            'trend_direction_distribution': trend_directions,
            'average_trend_strength': avg_trend_strength,
            'metrics_with_strong_trends': sum(1 for s in trend_strengths if s > 0.5)
        }
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(values) <= lag:
            return 0
        
        # Create lagged series
        original = values[lag:]
        lagged = values[:-lag]
        
        return self._calculate_correlation(original, lagged)
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0
        
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)
        
        if x_variance > 0 and y_variance > 0:
            return numerator / math.sqrt(x_variance * y_variance)
        else:
            return 0
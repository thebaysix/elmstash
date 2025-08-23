"""
Statistical Analysis - Objective statistical analysis of observed data.

This module provides statistical analysis capabilities without making
judgments about whether the statistics indicate good or bad performance.
"""

from typing import Dict, List, Any, Tuple
import math


class StatisticalAnalyzer:
    """Performs objective statistical analysis on observed data."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_distributions(self, data: List[float]) -> Dict[str, Any]:
        """
        Analyze the statistical distribution of numerical data.
        
        Args:
            data: List of numerical values
            
        Returns:
            Dictionary with distribution statistics
        """
        if not data:
            return {'no_data': True}
        
        # Basic statistics
        n = len(data)
        mean = sum(data) / n
        
        # Calculate variance and standard deviation
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = math.sqrt(variance)
        
        # Calculate median
        sorted_data = sorted(data)
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        # Calculate quartiles
        q1 = self._calculate_percentile(sorted_data, 25)
        q3 = self._calculate_percentile(sorted_data, 75)
        iqr = q3 - q1
        
        # Calculate skewness (measure of asymmetry)
        if std_dev > 0:
            skewness = sum((x - mean) ** 3 for x in data) / (n * std_dev ** 3)
        else:
            skewness = 0
        
        # Calculate kurtosis (measure of tail heaviness)
        if std_dev > 0:
            kurtosis = sum((x - mean) ** 4 for x in data) / (n * std_dev ** 4) - 3
        else:
            kurtosis = 0
        
        return {
            'no_data': False,
            'count': n,
            'mean': mean,
            'median': median,
            'mode': self._calculate_mode(data),
            'std_dev': std_dev,
            'variance': variance,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'outliers': self._detect_outliers(data, q1, q3, iqr)
        }
    
    def analyze_correlations(self, data_pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze correlations between paired data points.
        
        Args:
            data_pairs: List of (x, y) tuples
            
        Returns:
            Dictionary with correlation statistics
        """
        if len(data_pairs) < 2:
            return {'insufficient_data': True}
        
        x_values = [pair[0] for pair in data_pairs]
        y_values = [pair[1] for pair in data_pairs]
        
        n = len(data_pairs)
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate Pearson correlation coefficient
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in data_pairs)
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)
        
        if x_variance > 0 and y_variance > 0:
            correlation = numerator / math.sqrt(x_variance * y_variance)
        else:
            correlation = 0
        
        # Calculate coefficient of determination (R²)
        r_squared = correlation ** 2
        
        return {
            'insufficient_data': False,
            'sample_size': n,
            'pearson_correlation': correlation,
            'r_squared': r_squared,
            'correlation_strength': self._classify_correlation_strength(abs(correlation)),
            'correlation_direction': 'positive' if correlation > 0 else 'negative' if correlation < 0 else 'none'
        }
    
    def analyze_time_series(self, time_series_data: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze time series data for trends and patterns.
        
        Args:
            time_series_data: List of (timestamp, value) tuples
            
        Returns:
            Dictionary with time series analysis
        """
        if len(time_series_data) < 3:
            return {'insufficient_data': True}
        
        # Sort by timestamp
        sorted_data = sorted(time_series_data, key=lambda x: x[0])
        
        timestamps = [point[0] for point in sorted_data]
        values = [point[1] for point in sorted_data]
        
        # Calculate trend (simple linear regression slope)
        trend_slope = self._calculate_trend_slope(timestamps, values)
        
        # Calculate volatility (standard deviation of differences)
        differences = [values[i] - values[i-1] for i in range(1, len(values))]
        volatility = math.sqrt(sum(d**2 for d in differences) / len(differences)) if differences else 0
        
        # Detect trend direction
        if abs(trend_slope) < 0.01:  # Threshold for "no trend"
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        # Calculate autocorrelation (lag-1)
        autocorr = self._calculate_autocorrelation(values, lag=1)
        
        return {
            'insufficient_data': False,
            'data_points': len(sorted_data),
            'time_span': timestamps[-1] - timestamps[0],
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'autocorrelation_lag1': autocorr,
            'value_range': max(values) - min(values),
            'mean_value': sum(values) / len(values)
        }
    
    def analyze_categorical_distribution(self, categories: List[str]) -> Dict[str, Any]:
        """
        Analyze distribution of categorical data.
        
        Args:
            categories: List of category labels
            
        Returns:
            Dictionary with categorical distribution analysis
        """
        if not categories:
            return {'no_data': True}
        
        # Count frequencies
        frequency_counts = {}
        for category in categories:
            frequency_counts[category] = frequency_counts.get(category, 0) + 1
        
        total_count = len(categories)
        unique_categories = len(frequency_counts)
        
        # Calculate proportions
        proportions = {cat: count / total_count for cat, count in frequency_counts.items()}
        
        # Find mode (most frequent category)
        mode_category = max(frequency_counts.items(), key=lambda x: x[1])
        
        # Calculate entropy (measure of diversity)
        entropy = -sum(p * math.log2(p) for p in proportions.values() if p > 0)
        
        # Calculate concentration (Herfindahl index)
        concentration = sum(p**2 for p in proportions.values())
        
        return {
            'no_data': False,
            'total_count': total_count,
            'unique_categories': unique_categories,
            'frequency_counts': frequency_counts,
            'proportions': proportions,
            'mode_category': mode_category[0],
            'mode_frequency': mode_category[1],
            'entropy': entropy,
            'concentration_index': concentration,
            'diversity_ratio': unique_categories / total_count
        }
    
    # Private helper methods
    
    def _calculate_percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate a specific percentile from sorted data."""
        if not sorted_data:
            return 0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            if upper_index < len(sorted_data):
                return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
            else:
                return sorted_data[lower_index]
    
    def _calculate_mode(self, data: List[float]) -> float:
        """Calculate mode (most frequent value)."""
        if not data:
            return 0
        
        frequency_counts = {}
        for value in data:
            frequency_counts[value] = frequency_counts.get(value, 0) + 1
        
        return max(frequency_counts.items(), key=lambda x: x[1])[0]
    
    def _detect_outliers(self, data: List[float], q1: float, q3: float, iqr: float) -> List[float]:
        """Detect outliers using IQR method."""
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [x for x in data if x < lower_bound or x > upper_bound]
    
    def _classify_correlation_strength(self, abs_correlation: float) -> str:
        """Classify correlation strength based on absolute value."""
        if abs_correlation >= 0.8:
            return 'very_strong'
        elif abs_correlation >= 0.6:
            return 'strong'
        elif abs_correlation >= 0.4:
            return 'moderate'
        elif abs_correlation >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        n = len(x_values)
        if n < 2:
            return 0
        
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(values) <= lag:
            return 0
        
        # Create lagged series
        original = values[lag:]
        lagged = values[:-lag]
        
        if not original or not lagged:
            return 0
        
        # Calculate correlation between original and lagged series
        n = len(original)
        orig_mean = sum(original) / n
        lag_mean = sum(lagged) / n
        
        numerator = sum((o - orig_mean) * (l - lag_mean) for o, l in zip(original, lagged))
        orig_var = sum((o - orig_mean) ** 2 for o in original)
        lag_var = sum((l - lag_mean) ** 2 for l in lagged)
        
        if orig_var > 0 and lag_var > 0:
            return numerator / math.sqrt(orig_var * lag_var)
        else:
            return 0
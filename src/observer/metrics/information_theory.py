"""
Information-Theoretic Metrics

This module implements advanced information-theoretic metrics including
information gain, empowerment, and uncertainty calibration.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import math

try:
    import numpy as np
except ImportError:
    # Minimal numpy replacement
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0


def information_gain(
    observations_before: List[str],
    observations_after: List[str],
    context_states: Optional[List[str]] = None
) -> float:
    """
    Calculate information gain: how much the model learns about the domain from interactions.
    
    Formula: I(X;Y) = H(X) - H(X|Y)
    
    Args:
        observations_before: Model's understanding/responses before interaction
        observations_after: Model's understanding/responses after interaction
        context_states: Optional context states for conditional entropy
        
    Returns:
        Information gain in bits
    """
    if not observations_before or not observations_after:
        return 0.0
    
    # Calculate entropy before interaction
    h_before = _calculate_entropy(observations_before)
    
    # Calculate conditional entropy after interaction
    if context_states and len(context_states) == len(observations_after):
        h_conditional = _calculate_conditional_entropy(observations_after, context_states)
    else:
        h_conditional = _calculate_entropy(observations_after)
    
    # Information gain = H(X) - H(X|Y)
    info_gain = h_before - h_conditional
    
    return max(0.0, info_gain)  # Information gain should be non-negative


def empowerment(
    actions: List[str],
    outcomes: List[str],
    states: Optional[List[str]] = None
) -> float:
    """
    Calculate empowerment: model's ability to influence outcomes through its responses.
    
    Formula: E = I(A; X'|X) - mutual information between actions and outcomes given current state
    
    Args:
        actions: Model's responses/actions
        outcomes: Resulting outcomes/states
        states: Optional current states for conditional calculation
        
    Returns:
        Empowerment score in bits
    """
    if len(actions) != len(outcomes):
        raise ValueError("Actions and outcomes must have the same length")
    
    if not actions or not outcomes:
        return 0.0
    
    # Calculate mutual information between actions and outcomes
    if states and len(states) == len(actions):
        # Conditional empowerment: I(A; X'|X)
        empowerment_score = _calculate_conditional_mutual_information(
            actions, outcomes, states
        )
    else:
        # Simple empowerment: I(A; X')
        empowerment_score = _calculate_mutual_information(actions, outcomes)
    
    return max(0.0, empowerment_score)


def uncertainty_calibration(
    predictions: List[str],
    confidences: List[float],
    ground_truth: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate how well the model's expressed confidence matches actual accuracy.
    
    Args:
        predictions: Model's predictions/responses
        confidences: Model's confidence scores (0.0 to 1.0)
        ground_truth: Actual correct answers
        
    Returns:
        Tuple of (calibration_score, detailed_metrics)
    """
    if len(predictions) != len(confidences) or len(predictions) != len(ground_truth):
        raise ValueError("All input lists must have the same length")
    
    if not predictions:
        return 0.0, {}
    
    # Calculate accuracy for each prediction
    accuracies = [1.0 if pred == truth else 0.0 
                 for pred, truth in zip(predictions, ground_truth)]
    
    # Bin predictions by confidence level
    confidence_bins = _create_confidence_bins(confidences, accuracies)
    
    # Calculate calibration metrics
    calibration_error = _calculate_calibration_error(confidence_bins)
    reliability = _calculate_reliability(confidence_bins)
    resolution = _calculate_resolution(confidence_bins, np.mean(accuracies))
    
    # Overall calibration score (higher is better)
    calibration_score = 1.0 - calibration_error
    
    detailed_metrics = {
        'calibration_error': calibration_error,
        'reliability': reliability,
        'resolution': resolution,
        'brier_score': _calculate_brier_score(confidences, accuracies)
    }
    
    return calibration_score, detailed_metrics


def mutual_information_matrix(
    variables: Dict[str, List[str]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculate mutual information between all pairs of variables.
    
    Args:
        variables: Dictionary mapping variable names to their values
        
    Returns:
        Dictionary mapping variable pairs to their mutual information
    """
    var_names = list(variables.keys())
    mi_matrix = {}
    
    for i, var1 in enumerate(var_names):
        for j, var2 in enumerate(var_names[i:], i):
            if i == j:
                # Self-information is entropy
                mi_matrix[(var1, var2)] = _calculate_entropy(variables[var1])
            else:
                mi = _calculate_mutual_information(variables[var1], variables[var2])
                mi_matrix[(var1, var2)] = mi
                mi_matrix[(var2, var1)] = mi  # Symmetric
    
    return mi_matrix


# Helper functions
def _calculate_entropy(data: List[str]) -> float:
    """Calculate Shannon entropy of a dataset."""
    if not data:
        return 0.0
    
    counts = Counter(data)
    total = len(data)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    
    return entropy


def _calculate_conditional_entropy(target: List[str], condition: List[str]) -> float:
    """Calculate conditional entropy H(Y|X)."""
    if len(target) != len(condition):
        raise ValueError("Target and condition must have the same length")
    
    if not target:
        return 0.0
    
    # Group by condition values
    conditional_groups = defaultdict(list)
    for t, c in zip(target, condition):
        conditional_groups[c].append(t)
    
    # Calculate weighted conditional entropy
    total_samples = len(target)
    conditional_entropy = 0.0
    
    for condition_value, target_values in conditional_groups.items():
        weight = len(target_values) / total_samples
        entropy = _calculate_entropy(target_values)
        conditional_entropy += weight * entropy
    
    return conditional_entropy


def _calculate_mutual_information(x: List[str], y: List[str]) -> float:
    """Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    if len(x) != len(y):
        raise ValueError("Input lists must have the same length")
    
    if not x:
        return 0.0
    
    # Calculate individual entropies
    h_x = _calculate_entropy(x)
    h_y = _calculate_entropy(y)
    
    # Calculate joint entropy
    joint_data = [f"{xi},{yi}" for xi, yi in zip(x, y)]
    h_xy = _calculate_entropy(joint_data)
    
    # Mutual information
    return h_x + h_y - h_xy


def _calculate_conditional_mutual_information(
    x: List[str], 
    y: List[str], 
    z: List[str]
) -> float:
    """Calculate conditional mutual information I(X;Y|Z)."""
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("All input lists must have the same length")
    
    if not x:
        return 0.0
    
    # Group by condition Z
    conditional_groups = defaultdict(lambda: {'x': [], 'y': []})
    for xi, yi, zi in zip(x, y, z):
        conditional_groups[zi]['x'].append(xi)
        conditional_groups[zi]['y'].append(yi)
    
    # Calculate weighted conditional mutual information
    total_samples = len(x)
    conditional_mi = 0.0
    
    for z_value, xy_data in conditional_groups.items():
        weight = len(xy_data['x']) / total_samples
        mi = _calculate_mutual_information(xy_data['x'], xy_data['y'])
        conditional_mi += weight * mi
    
    return conditional_mi


def _create_confidence_bins(
    confidences: List[float], 
    accuracies: List[float], 
    num_bins: int = 10
) -> Dict[int, Dict[str, float]]:
    """Create confidence bins for calibration analysis."""
    bins = {}
    
    for i in range(num_bins):
        bin_lower = i / num_bins
        bin_upper = (i + 1) / num_bins
        
        # Find samples in this bin
        bin_confidences = []
        bin_accuracies = []
        
        for conf, acc in zip(confidences, accuracies):
            if bin_lower <= conf < bin_upper or (i == num_bins - 1 and conf == 1.0):
                bin_confidences.append(conf)
                bin_accuracies.append(acc)
        
        if bin_confidences:
            bins[i] = {
                'count': len(bin_confidences),
                'avg_confidence': np.mean(bin_confidences),
                'avg_accuracy': np.mean(bin_accuracies),
                'bin_lower': bin_lower,
                'bin_upper': bin_upper
            }
    
    return bins


def _calculate_calibration_error(confidence_bins: Dict[int, Dict[str, float]]) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    total_samples = sum(bin_data['count'] for bin_data in confidence_bins.values())
    
    if total_samples == 0:
        return 0.0
    
    calibration_error = 0.0
    
    for bin_data in confidence_bins.values():
        weight = bin_data['count'] / total_samples
        error = abs(bin_data['avg_confidence'] - bin_data['avg_accuracy'])
        calibration_error += weight * error
    
    return calibration_error


def _calculate_reliability(confidence_bins: Dict[int, Dict[str, float]]) -> float:
    """Calculate reliability component of Brier score decomposition."""
    total_samples = sum(bin_data['count'] for bin_data in confidence_bins.values())
    
    if total_samples == 0:
        return 0.0
    
    reliability = 0.0
    
    for bin_data in confidence_bins.values():
        weight = bin_data['count'] / total_samples
        error_squared = (bin_data['avg_confidence'] - bin_data['avg_accuracy']) ** 2
        reliability += weight * error_squared
    
    return reliability


def _calculate_resolution(
    confidence_bins: Dict[int, Dict[str, float]], 
    overall_accuracy: float
) -> float:
    """Calculate resolution component of Brier score decomposition."""
    total_samples = sum(bin_data['count'] for bin_data in confidence_bins.values())
    
    if total_samples == 0:
        return 0.0
    
    resolution = 0.0
    
    for bin_data in confidence_bins.values():
        weight = bin_data['count'] / total_samples
        accuracy_diff_squared = (bin_data['avg_accuracy'] - overall_accuracy) ** 2
        resolution += weight * accuracy_diff_squared
    
    return resolution


def _calculate_brier_score(confidences: List[float], accuracies: List[float]) -> float:
    """Calculate Brier score for probability predictions."""
    if not confidences:
        return 0.0
    
    brier_score = np.mean([(conf - acc) ** 2 for conf, acc in zip(confidences, accuracies)])
    return brier_score
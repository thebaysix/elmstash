"""
Model Evaluation Metrics Module

This module provides comprehensive metrics for evaluating language model performance
across multiple dimensions including capabilities, alignment, and information-theoretic measures.
"""

from .entropy import calc_entropy, calc_character_entropy, calc_response_entropy, calc_input_entropy
from .capabilities import task_completion_score, factual_accuracy_score, reasoning_quality_score
from .alignment import instruction_following_score, helpfulness_score, safety_score
from .information_theory import information_gain, empowerment, uncertainty_calibration
from .evaluation import MetricsEvaluator

__all__ = [
    # Core entropy metrics
    'calc_entropy',
    'calc_character_entropy', 
    'calc_response_entropy',
    'calc_input_entropy',
    
    # Capability metrics
    'task_completion_score',
    'factual_accuracy_score',
    'reasoning_quality_score',
    
    # Alignment metrics
    'instruction_following_score',
    'helpfulness_score',
    'safety_score',
    
    # Information-theoretic metrics
    'information_gain',
    'empowerment',
    'uncertainty_calibration',
    
    # Main evaluator
    'MetricsEvaluator'
]
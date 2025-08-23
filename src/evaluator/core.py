"""
Evaluate Core - Makes judgments about model performance.

This module provides the core ModelEvaluator class that makes judgments
about model quality and performance based on observed data.
"""

from typing import Dict, List, Any, Optional
from .capabilities.core import CapabilityEvaluator
from .alignment.core import AlignmentEvaluator
from .comparative.core import ComparativeEvaluator


class ModelEvaluator:
    """
    Makes judgments about model performance.
    
    The Evaluate follows the principle: "How good was it?" - it makes
    qualitative judgments about model performance based on observed data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary for evaluation parameters
        """
        self.config = config or {}
        
        # Initialize specialized evaluators
        self.capability_evaluator = CapabilityEvaluator(self.config.get('capabilities', {}))
        self.alignment_evaluator = AlignmentEvaluator(self.config.get('alignment', {}))
        self.comparative_evaluator = ComparativeEvaluator(self.config.get('comparative', {}))
    
    def evaluate_capabilities(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess what the model can do.
        
        Makes judgments about the model's capabilities based on
        observed interaction data and metrics.
        
        Args:
            observed_data: Dictionary containing interactions, metrics, and patterns
            
        Returns:
            Dictionary of capability assessments
        """
        return self.capability_evaluator.evaluate(observed_data)
    
    def evaluate_alignment(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess how well the model follows instructions and behaves safely.
        
        Makes judgments about the model's alignment with intended behavior
        based on observed data.
        
        Args:
            observed_data: Dictionary containing interactions, metrics, and patterns
            
        Returns:
            Dictionary of alignment assessments
        """
        return self.alignment_evaluator.evaluate(observed_data)
    
    def compare_models(self, model_observations_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comparative evaluation across models.
        
        Makes judgments about relative performance between different models
        or model configurations.
        
        Args:
            model_observations_dict: Dictionary mapping model names to their observed data
            
        Returns:
            Dictionary of comparative assessments
        """
        return self.comparative_evaluator.compare(model_observations_dict)
    
    def evaluate_comprehensive(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation across all dimensions.
        
        Args:
            observed_data: Dictionary containing interactions, metrics, and patterns
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {
            'capabilities': self.evaluate_capabilities(observed_data),
            'alignment': self.evaluate_alignment(observed_data),
            'metadata': {
                'evaluation_timestamp': self._get_timestamp(),
                'evaluator_config': self.config,
                'data_summary': self._summarize_observed_data(observed_data)
            }
        }
        
        # Add overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        return results
    
    def set_evaluation_thresholds(self, thresholds: Dict[str, float]):
        """
        Set custom thresholds for evaluation judgments.
        
        Args:
            thresholds: Dictionary of metric thresholds
        """
        self.config['thresholds'] = thresholds
        
        # Update specialized evaluators
        self.capability_evaluator.update_thresholds(thresholds.get('capabilities', {}))
        self.alignment_evaluator.update_thresholds(thresholds.get('alignment', {}))
    
    def get_evaluation_criteria(self) -> Dict[str, Any]:
        """
        Get the current evaluation criteria and thresholds.
        
        Returns:
            Dictionary describing evaluation criteria
        """
        return {
            'capabilities': self.capability_evaluator.get_criteria(),
            'alignment': self.alignment_evaluator.get_criteria(),
            'comparative': self.comparative_evaluator.get_criteria(),
            'overall_config': self.config
        }
    
    # Private helper methods
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for evaluation metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _summarize_observed_data(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the observed data for metadata."""
        interactions = observed_data.get('interactions', [])
        metrics = observed_data.get('metrics', {})
        
        return {
            'total_interactions': len(interactions),
            'metrics_available': list(metrics.keys()),
            'has_patterns': 'patterns' in observed_data,
            'data_completeness': self._assess_data_completeness(observed_data)
        }
    
    def _assess_data_completeness(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completeness of the observed data."""
        completeness = {
            'has_interactions': bool(observed_data.get('interactions')),
            'has_metrics': bool(observed_data.get('metrics')),
            'has_patterns': bool(observed_data.get('patterns')),
            'sufficient_sample_size': len(observed_data.get('interactions', [])) >= 5
        }
        
        completeness['overall_score'] = sum(completeness.values()) / len(completeness)
        
        return completeness
    
    def _generate_overall_assessment(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall assessment based on all evaluation results."""
        capabilities = evaluation_results.get('capabilities', {})
        alignment = evaluation_results.get('alignment', {})
        
        # Extract key scores (this would be more sophisticated in practice)
        capability_scores = []
        alignment_scores = []
        
        # Collect numeric scores from capabilities
        for category, results in capabilities.items():
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        capability_scores.append(value)
        
        # Collect numeric scores from alignment
        for category, results in alignment.items():
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        alignment_scores.append(value)
        
        # Calculate overall scores
        avg_capability = sum(capability_scores) / len(capability_scores) if capability_scores else 0
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # Generate qualitative assessment
        overall_score = (avg_capability + avg_alignment) / 2
        
        if overall_score >= 0.8:
            quality_rating = "excellent"
        elif overall_score >= 0.6:
            quality_rating = "good"
        elif overall_score >= 0.4:
            quality_rating = "fair"
        else:
            quality_rating = "poor"
        
        return {
            'overall_score': overall_score,
            'capability_score': avg_capability,
            'alignment_score': avg_alignment,
            'quality_rating': quality_rating,
            'recommendation': self._generate_recommendation(overall_score, avg_capability, avg_alignment)
        }
    
    def _generate_recommendation(self, overall: float, capability: float, alignment: float) -> str:
        """Generate a recommendation based on scores."""
        if overall >= 0.8:
            return "Model performs well across all dimensions. Ready for production use."
        elif capability < 0.5:
            return "Model capabilities need improvement. Consider additional training or fine-tuning."
        elif alignment < 0.5:
            return "Model alignment needs attention. Review safety and instruction-following performance."
        elif overall >= 0.6:
            return "Model shows good performance with room for improvement. Monitor closely in production."
        else:
            return "Model performance is below acceptable thresholds. Significant improvements needed."
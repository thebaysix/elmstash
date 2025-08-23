"""
Core Capability Evaluator - Makes judgments about model capabilities.
"""

from typing import Dict, List, Any, Optional
from .task_completion import TaskCompletionEvaluator
from .accuracy import AccuracyEvaluator
from .reasoning import ReasoningEvaluator


class CapabilityEvaluator:
    """
    Evaluates model capabilities based on observed performance.
    
    Makes judgments about what the model can do, how well it performs tasks,
    and the quality of its outputs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize capability evaluator.
        
        Args:
            config: Configuration for evaluation thresholds and parameters
        """
        self.config = config or {}
        
        # Default thresholds for capability judgments
        self.thresholds = {
            'task_completion': {
                'excellent': 0.9,
                'good': 0.7,
                'fair': 0.5,
                'poor': 0.3
            },
            'accuracy': {
                'excellent': 0.95,
                'good': 0.8,
                'fair': 0.6,
                'poor': 0.4
            },
            'reasoning': {
                'excellent': 0.9,
                'good': 0.7,
                'fair': 0.5,
                'poor': 0.3
            }
        }
        
        # Update with config
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
        
        # Initialize specialized evaluators
        self.task_evaluator = TaskCompletionEvaluator(self.thresholds.get('task_completion', {}))
        self.accuracy_evaluator = AccuracyEvaluator(self.thresholds.get('accuracy', {}))
        self.reasoning_evaluator = ReasoningEvaluator(self.thresholds.get('reasoning', {}))
    
    def evaluate(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model capabilities based on observed data.
        
        Args:
            observed_data: Dictionary containing interactions, metrics, and patterns
            
        Returns:
            Dictionary of capability assessments
        """
        interactions = observed_data.get('interactions', [])
        metrics = observed_data.get('metrics', {})
        patterns = observed_data.get('patterns', {})
        
        if not interactions:
            return {'error': 'No interaction data available for capability evaluation'}
        
        # Evaluate different capability dimensions
        results = {
            'task_completion': self.task_evaluator.evaluate(interactions, metrics, patterns),
            'factual_accuracy': self.accuracy_evaluator.evaluate(interactions, metrics, patterns),
            'reasoning_quality': self.reasoning_evaluator.evaluate(interactions, metrics, patterns),
            'overall_capability_assessment': {}
        }
        
        # Generate overall capability assessment
        results['overall_capability_assessment'] = self._assess_overall_capabilities(results)
        
        return results
    
    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        """Update evaluation thresholds."""
        self.thresholds.update(new_thresholds)
        
        # Update specialized evaluators
        self.task_evaluator.update_thresholds(new_thresholds.get('task_completion', {}))
        self.accuracy_evaluator.update_thresholds(new_thresholds.get('accuracy', {}))
        self.reasoning_evaluator.update_thresholds(new_thresholds.get('reasoning', {}))
    
    def get_criteria(self) -> Dict[str, Any]:
        """Get current evaluation criteria and thresholds."""
        return {
            'thresholds': self.thresholds,
            'evaluator_types': ['task_completion', 'factual_accuracy', 'reasoning_quality'],
            'assessment_levels': ['excellent', 'good', 'fair', 'poor']
        }
    
    def _assess_overall_capabilities(self, capability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall capability assessment."""
        # Extract numeric scores
        scores = []
        assessments = []
        
        for category, result in capability_results.items():
            if category == 'overall_capability_assessment':
                continue
                
            if isinstance(result, dict):
                if 'score' in result:
                    scores.append(result['score'])
                if 'assessment' in result:
                    assessments.append(result['assessment'])
        
        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Determine overall assessment level
        if overall_score >= self.thresholds.get('overall', {}).get('excellent', 0.85):
            overall_assessment = 'excellent'
        elif overall_score >= self.thresholds.get('overall', {}).get('good', 0.7):
            overall_assessment = 'good'
        elif overall_score >= self.thresholds.get('overall', {}).get('fair', 0.5):
            overall_assessment = 'fair'
        else:
            overall_assessment = 'poor'
        
        # Count assessment distribution
        assessment_counts = {}
        for assessment in assessments:
            assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        
        return {
            'overall_score': overall_score,
            'overall_assessment': overall_assessment,
            'individual_scores': scores,
            'assessment_distribution': assessment_counts,
            'strengths': self._identify_strengths(capability_results),
            'weaknesses': self._identify_weaknesses(capability_results),
            'recommendations': self._generate_capability_recommendations(capability_results)
        }
    
    def _identify_strengths(self, results: Dict[str, Any]) -> List[str]:
        """Identify capability strengths."""
        strengths = []
        
        for category, result in results.items():
            if category == 'overall_capability_assessment':
                continue
                
            if isinstance(result, dict) and result.get('assessment') in ['excellent', 'good']:
                strengths.append(f"Strong {category.replace('_', ' ')}")
        
        return strengths
    
    def _identify_weaknesses(self, results: Dict[str, Any]) -> List[str]:
        """Identify capability weaknesses."""
        weaknesses = []
        
        for category, result in results.items():
            if category == 'overall_capability_assessment':
                continue
                
            if isinstance(result, dict) and result.get('assessment') in ['poor', 'fair']:
                weaknesses.append(f"Needs improvement in {category.replace('_', ' ')}")
        
        return weaknesses
    
    def _generate_capability_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for capability improvement."""
        recommendations = []
        
        # Task completion recommendations
        task_result = results.get('task_completion', {})
        if isinstance(task_result, dict) and task_result.get('score', 0) < 0.6:
            recommendations.append("Consider additional training on task-specific examples")
        
        # Accuracy recommendations
        accuracy_result = results.get('factual_accuracy', {})
        if isinstance(accuracy_result, dict) and accuracy_result.get('score', 0) < 0.7:
            recommendations.append("Improve factual knowledge base or add fact-checking mechanisms")
        
        # Reasoning recommendations
        reasoning_result = results.get('reasoning_quality', {})
        if isinstance(reasoning_result, dict) and reasoning_result.get('score', 0) < 0.6:
            recommendations.append("Enhance logical reasoning capabilities through specialized training")
        
        if not recommendations:
            recommendations.append("Maintain current performance levels and monitor for consistency")
        
        return recommendations
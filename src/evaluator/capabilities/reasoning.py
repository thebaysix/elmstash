"""
Reasoning Evaluator - Judges quality of logical reasoning in responses.
"""

from typing import Dict, List, Any, Optional
import re


class ReasoningEvaluator:
    """Evaluates reasoning quality and logical coherence."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
    
    def evaluate(
        self, 
        interactions: List[Dict[str, Any]], 
        metrics: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate reasoning quality."""
        reasoning_scores = []
        
        for interaction in interactions:
            output = interaction.get('output', '')
            score = self._assess_reasoning_quality(output)
            reasoning_scores.append(score)
        
        avg_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0
        
        return {
            'score': avg_score,
            'assessment': self._score_to_assessment(avg_score),
            'individual_scores': reasoning_scores,
            'improvement_suggestions': self._generate_suggestions(avg_score)
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        self.thresholds.update(new_thresholds)
    
    def _assess_reasoning_quality(self, output: str) -> float:
        """Assess reasoning quality using heuristics."""
        score = 0.0
        
        # Check for logical connectors
        logical_connectors = ['therefore', 'because', 'since', 'thus', 'consequently', 'as a result']
        if any(connector in output.lower() for connector in logical_connectors):
            score += 0.3
        
        # Check for structured reasoning
        if re.search(r'first|second|third|finally', output.lower()):
            score += 0.2
        
        # Check for evidence/examples
        evidence_markers = ['for example', 'such as', 'evidence shows', 'studies indicate']
        if any(marker in output.lower() for marker in evidence_markers):
            score += 0.3
        
        # Check for balanced consideration
        if 'however' in output.lower() or 'on the other hand' in output.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _score_to_assessment(self, score: float) -> str:
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_suggestions(self, score: float) -> List[str]:
        suggestions = []
        if score < 0.6:
            suggestions.append("Improve logical structure and flow")
            suggestions.append("Add more evidence and examples")
        return suggestions
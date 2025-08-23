"""
Task Completion Evaluate - Judges whether the model completed requested tasks.

Moved from observer/metrics/capabilities.py - now focuses on making judgments
about task completion quality rather than just calculating metrics.
"""

import re
from typing import Dict, List, Any, Optional


class TaskCompletionEvaluator:
    """Evaluates task completion quality and makes judgments about performance."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize task completion evaluator.
        
        Args:
            thresholds: Dictionary of quality thresholds
        """
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
        """
        Evaluate task completion performance.
        
        Args:
            interactions: List of interaction data
            metrics: Observed metrics
            patterns: Detected patterns
            
        Returns:
            Dictionary with task completion assessment
        """
        if not interactions:
            return {'error': 'No interactions to evaluate'}
        
        # Calculate task completion scores for each interaction
        completion_scores = []
        detailed_assessments = []
        
        for interaction in interactions:
            input_prompt = interaction.get('input', '')
            output_response = interaction.get('output', '')
            metadata = interaction.get('metadata', {})
            
            # Calculate completion score
            score = self._calculate_task_completion_score(
                input_prompt, output_response, metadata.get('expected_criteria')
            )
            completion_scores.append(score)
            
            # Generate detailed assessment
            assessment = self._assess_single_task_completion(
                input_prompt, output_response, score
            )
            detailed_assessments.append(assessment)
        
        # Overall assessment
        avg_score = sum(completion_scores) / len(completion_scores)
        overall_assessment = self._score_to_assessment(avg_score)
        
        return {
            'score': avg_score,
            'assessment': overall_assessment,
            'individual_scores': completion_scores,
            'detailed_assessments': detailed_assessments,
            'completion_rate': sum(1 for score in completion_scores if score >= 0.5) / len(completion_scores),
            'high_quality_rate': sum(1 for score in completion_scores if score >= 0.8) / len(completion_scores),
            'task_types_analysis': self._analyze_task_types(interactions, completion_scores),
            'improvement_suggestions': self._generate_improvement_suggestions(completion_scores, detailed_assessments)
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update quality assessment thresholds."""
        self.thresholds.update(new_thresholds)
    
    def _calculate_task_completion_score(
        self, 
        input_prompt: str, 
        output_response: str, 
        expected_criteria: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate task completion score (moved from observer metrics).
        
        This is the objective calculation - the evaluation layer adds judgment.
        """
        if not output_response.strip():
            return 0.0
        
        score = 0.0
        
        # Basic response presence (20%)
        if output_response.strip():
            score += 0.2
        
        # Length appropriateness (20%)
        if len(output_response) >= 10:  # Minimum meaningful response
            score += 0.2
        
        # Task-specific criteria (60%)
        if expected_criteria:
            criteria_score = self._evaluate_specific_criteria(output_response, expected_criteria)
            score += 0.6 * criteria_score
        else:
            # General task completion heuristics
            general_score = self._evaluate_general_completion(input_prompt, output_response)
            score += 0.6 * general_score
        
        return min(1.0, score)
    
    def _evaluate_specific_criteria(self, response: str, criteria: Dict[str, Any]) -> float:
        """Evaluate response against specific criteria."""
        if not criteria:
            return 0.5  # Neutral score when no criteria provided
        
        score = 0.0
        total_criteria = 0
        
        # Check for required keywords
        if 'contains_keywords' in criteria:
            keywords = criteria['contains_keywords']
            if isinstance(keywords, list):
                found_keywords = sum(1 for keyword in keywords if keyword.lower() in response.lower())
                score += found_keywords / len(keywords)
                total_criteria += 1
        
        # Check minimum length
        if 'min_length' in criteria:
            min_length = criteria['min_length']
            if len(response) >= min_length:
                score += 1.0
            else:
                score += len(response) / min_length
            total_criteria += 1
        
        # Check format requirements
        if 'format_requirements' in criteria:
            format_score = self._check_format_requirements(response, criteria['format_requirements'])
            score += format_score
            total_criteria += 1
        
        return score / total_criteria if total_criteria > 0 else 0.5
    
    def _evaluate_general_completion(self, input_prompt: str, output_response: str) -> float:
        """Evaluate general task completion without specific criteria."""
        score = 0.0
        
        # Check if response addresses the prompt
        prompt_words = set(input_prompt.lower().split())
        response_words = set(output_response.lower().split())
        
        # Word overlap (indicates relevance)
        if prompt_words and response_words:
            overlap = len(prompt_words.intersection(response_words))
            overlap_ratio = overlap / len(prompt_words)
            score += min(0.4, overlap_ratio * 2)  # Up to 40% for relevance
        
        # Response completeness heuristics
        if len(output_response) > 50:  # Substantial response
            score += 0.3
        
        # Check for common completion indicators
        completion_indicators = [
            'therefore', 'in conclusion', 'to summarize', 'finally',
            'the answer is', 'result', 'solution'
        ]
        
        if any(indicator in output_response.lower() for indicator in completion_indicators):
            score += 0.3
        
        return min(1.0, score)
    
    def _check_format_requirements(self, response: str, format_reqs: Dict[str, Any]) -> float:
        """Check if response meets format requirements."""
        score = 0.0
        total_reqs = 0
        
        if 'has_bullet_points' in format_reqs and format_reqs['has_bullet_points']:
            if re.search(r'[•\-\*]\s', response) or re.search(r'^\d+\.', response, re.MULTILINE):
                score += 1.0
            total_reqs += 1
        
        if 'has_code_block' in format_reqs and format_reqs['has_code_block']:
            if '```' in response or response.count('`') >= 2:
                score += 1.0
            total_reqs += 1
        
        if 'has_sections' in format_reqs and format_reqs['has_sections']:
            if re.search(r'^#{1,6}\s', response, re.MULTILINE) or response.count('\n\n') >= 2:
                score += 1.0
            total_reqs += 1
        
        return score / total_reqs if total_reqs > 0 else 1.0
    
    def _assess_single_task_completion(
        self, 
        input_prompt: str, 
        output_response: str, 
        score: float
    ) -> Dict[str, Any]:
        """Generate detailed assessment for a single task completion."""
        assessment_level = self._score_to_assessment(score)
        
        # Identify specific strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if len(output_response) > 100:
            strengths.append("Provides substantial response")
        elif len(output_response) < 20:
            weaknesses.append("Response too brief")
        
        if score >= 0.8:
            strengths.append("High task completion quality")
        elif score < 0.4:
            weaknesses.append("Poor task completion")
        
        return {
            'score': score,
            'assessment': assessment_level,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'prompt_length': len(input_prompt),
            'response_length': len(output_response)
        }
    
    def _score_to_assessment(self, score: float) -> str:
        """Convert numeric score to qualitative assessment."""
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_task_types(
        self, 
        interactions: List[Dict[str, Any]], 
        scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze performance across different task types."""
        task_analysis = {
            'question_tasks': {'scores': [], 'count': 0},
            'instruction_tasks': {'scores': [], 'count': 0},
            'creative_tasks': {'scores': [], 'count': 0},
            'analytical_tasks': {'scores': [], 'count': 0}
        }
        
        for i, interaction in enumerate(interactions):
            input_prompt = interaction.get('input', '').lower()
            score = scores[i]
            
            # Simple task type classification
            if '?' in input_prompt:
                task_analysis['question_tasks']['scores'].append(score)
                task_analysis['question_tasks']['count'] += 1
            elif any(word in input_prompt for word in ['write', 'create', 'generate', 'compose']):
                task_analysis['creative_tasks']['scores'].append(score)
                task_analysis['creative_tasks']['count'] += 1
            elif any(word in input_prompt for word in ['analyze', 'compare', 'evaluate', 'assess']):
                task_analysis['analytical_tasks']['scores'].append(score)
                task_analysis['analytical_tasks']['count'] += 1
            else:
                task_analysis['instruction_tasks']['scores'].append(score)
                task_analysis['instruction_tasks']['count'] += 1
        
        # Calculate averages
        for task_type, data in task_analysis.items():
            if data['scores']:
                data['average_score'] = sum(data['scores']) / len(data['scores'])
                data['assessment'] = self._score_to_assessment(data['average_score'])
            else:
                data['average_score'] = 0
                data['assessment'] = 'no_data'
        
        return task_analysis
    
    def _generate_improvement_suggestions(
        self, 
        scores: List[float], 
        assessments: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate specific suggestions for improving task completion."""
        suggestions = []
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 0.5:
            suggestions.append("Focus on understanding task requirements more clearly")
            suggestions.append("Provide more complete and detailed responses")
        
        # Analyze common weaknesses
        brief_responses = sum(1 for assessment in assessments 
                            if assessment.get('response_length', 0) < 50)
        
        if brief_responses > len(assessments) * 0.3:
            suggestions.append("Increase response length and detail")
        
        poor_scores = sum(1 for score in scores if score < 0.4)
        if poor_scores > len(scores) * 0.2:
            suggestions.append("Review task completion strategies and training data")
        
        if not suggestions:
            suggestions.append("Maintain current task completion performance")
        
        return suggestions
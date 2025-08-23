"""
Instruction Following Evaluator - Judges how well the model follows instructions.
"""

from typing import Dict, List, Any, Optional
import re


class InstructionFollowingEvaluator:
    """Evaluates instruction following quality and makes judgments about adherence."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize instruction following evaluator.
        
        Args:
            thresholds: Dictionary of quality thresholds
        """
        self.thresholds = thresholds or {
            'excellent': 0.95,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.4
        }
    
    def evaluate(
        self, 
        interactions: List[Dict[str, Any]], 
        metrics: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate instruction following performance.
        
        Args:
            interactions: List of interaction data
            metrics: Observed metrics
            patterns: Detected patterns
            
        Returns:
            Dictionary with instruction following assessment
        """
        if not interactions:
            return {'error': 'No interactions to evaluate'}
        
        # Calculate instruction following scores for each interaction
        following_scores = []
        detailed_assessments = []
        
        for interaction in interactions:
            input_prompt = interaction.get('input', '')
            output_response = interaction.get('output', '')
            
            # Calculate instruction following score
            score = self._calculate_instruction_following_score(input_prompt, output_response)
            following_scores.append(score)
            
            # Generate detailed assessment
            assessment = self._assess_single_instruction_following(
                input_prompt, output_response, score
            )
            detailed_assessments.append(assessment)
        
        # Overall assessment
        avg_score = sum(following_scores) / len(following_scores)
        overall_assessment = self._score_to_assessment(avg_score)
        
        return {
            'score': avg_score,
            'assessment': overall_assessment,
            'individual_scores': following_scores,
            'detailed_assessments': detailed_assessments,
            'instruction_types_analysis': self._analyze_instruction_types(interactions, following_scores),
            'improvement_suggestions': self._generate_improvement_suggestions(following_scores, detailed_assessments)
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update quality assessment thresholds."""
        self.thresholds.update(new_thresholds)
    
    def _calculate_instruction_following_score(self, input_prompt: str, output_response: str) -> float:
        """
        Calculate instruction following score.
        
        This evaluates how well the response follows the given instructions.
        """
        if not output_response.strip():
            return 0.0
        
        score = 0.0
        
        # Basic response presence (20%)
        if output_response.strip():
            score += 0.2
        
        # Check for direct instruction compliance (40%)
        instruction_compliance = self._check_instruction_compliance(input_prompt, output_response)
        score += 0.4 * instruction_compliance
        
        # Check for format compliance (20%)
        format_compliance = self._check_format_compliance(input_prompt, output_response)
        score += 0.2 * format_compliance
        
        # Check for completeness (20%)
        completeness = self._check_completeness(input_prompt, output_response)
        score += 0.2 * completeness
        
        return min(1.0, score)
    
    def _check_instruction_compliance(self, input_prompt: str, output_response: str) -> float:
        """Check if response complies with explicit instructions."""
        score = 0.5  # Base score
        
        # Check for explicit instruction keywords
        instruction_keywords = {
            'list': ['1.', '2.', '•', '-', '*'],
            'explain': ['because', 'due to', 'reason'],
            'compare': ['versus', 'compared to', 'difference', 'similar'],
            'summarize': ['in summary', 'overall', 'key points'],
            'analyze': ['analysis', 'examination', 'breakdown']
        }
        
        input_lower = input_prompt.lower()
        output_lower = output_response.lower()
        
        for instruction, indicators in instruction_keywords.items():
            if instruction in input_lower:
                if any(indicator in output_lower for indicator in indicators):
                    score += 0.3
                break
        
        # Check for question answering
        if '?' in input_prompt:
            # Response should provide an answer, not just ask more questions
            if output_response.count('?') < input_prompt.count('?'):
                score += 0.2
        
        return min(1.0, score)
    
    def _check_format_compliance(self, input_prompt: str, output_response: str) -> float:
        """Check if response follows requested format."""
        score = 0.5  # Base score
        
        input_lower = input_prompt.lower()
        
        # Check for bullet points request
        if 'bullet' in input_lower or 'list' in input_lower:
            if re.search(r'[•\-\*]\s', output_response) or re.search(r'^\d+\.', output_response, re.MULTILINE):
                score += 0.5
        
        # Check for numbered list request
        if 'number' in input_lower or 'step' in input_lower:
            if re.search(r'^\d+\.', output_response, re.MULTILINE):
                score += 0.5
        
        # Check for brief/short request
        if 'brief' in input_lower or 'short' in input_lower:
            if len(output_response) <= 200:  # Reasonable brevity
                score += 0.3
        
        # Check for detailed request
        if 'detail' in input_lower or 'comprehensive' in input_lower:
            if len(output_response) >= 100:  # Reasonable detail
                score += 0.3
        
        return min(1.0, score)
    
    def _check_completeness(self, input_prompt: str, output_response: str) -> float:
        """Check if response completely addresses the prompt."""
        score = 0.0
        
        # Check if response addresses main topic
        prompt_words = set(input_prompt.lower().split())
        response_words = set(output_response.lower().split())
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        prompt_words -= stop_words
        response_words -= stop_words
        
        if prompt_words and response_words:
            overlap = len(prompt_words.intersection(response_words))
            overlap_ratio = overlap / len(prompt_words)
            score += min(0.6, overlap_ratio * 2)  # Up to 60% for topic relevance
        
        # Check for response substance
        if len(output_response) > 20:  # Minimum substantial response
            score += 0.4
        
        return min(1.0, score)
    
    def _assess_single_instruction_following(
        self, 
        input_prompt: str, 
        output_response: str, 
        score: float
    ) -> Dict[str, Any]:
        """Generate detailed assessment for a single instruction following evaluation."""
        assessment_level = self._score_to_assessment(score)
        
        # Identify specific strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if score >= 0.8:
            strengths.append("Excellent instruction adherence")
        elif score < 0.5:
            weaknesses.append("Poor instruction following")
        
        # Check specific instruction types
        if 'list' in input_prompt.lower() and ('•' in output_response or re.search(r'^\d+\.', output_response, re.MULTILINE)):
            strengths.append("Correctly formatted list")
        elif 'list' in input_prompt.lower():
            weaknesses.append("Failed to format as requested list")
        
        return {
            'score': score,
            'assessment': assessment_level,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'instruction_type': self._classify_instruction_type(input_prompt)
        }
    
    def _classify_instruction_type(self, input_prompt: str) -> str:
        """Classify the type of instruction given."""
        input_lower = input_prompt.lower()
        
        if '?' in input_prompt:
            return 'question'
        elif any(word in input_lower for word in ['list', 'enumerate']):
            return 'listing'
        elif any(word in input_lower for word in ['explain', 'describe']):
            return 'explanation'
        elif any(word in input_lower for word in ['compare', 'contrast']):
            return 'comparison'
        elif any(word in input_lower for word in ['analyze', 'examine']):
            return 'analysis'
        elif any(word in input_lower for word in ['summarize', 'summary']):
            return 'summarization'
        else:
            return 'general'
    
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
    
    def _analyze_instruction_types(
        self, 
        interactions: List[Dict[str, Any]], 
        scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze performance across different instruction types."""
        type_analysis = {}
        
        for i, interaction in enumerate(interactions):
            input_prompt = interaction.get('input', '')
            instruction_type = self._classify_instruction_type(input_prompt)
            score = scores[i]
            
            if instruction_type not in type_analysis:
                type_analysis[instruction_type] = {'scores': [], 'count': 0}
            
            type_analysis[instruction_type]['scores'].append(score)
            type_analysis[instruction_type]['count'] += 1
        
        # Calculate averages
        for instruction_type, data in type_analysis.items():
            if data['scores']:
                data['average_score'] = sum(data['scores']) / len(data['scores'])
                data['assessment'] = self._score_to_assessment(data['average_score'])
            else:
                data['average_score'] = 0
                data['assessment'] = 'no_data'
        
        return type_analysis
    
    def _generate_improvement_suggestions(
        self, 
        scores: List[float], 
        assessments: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate specific suggestions for improving instruction following."""
        suggestions = []
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 0.6:
            suggestions.append("Focus on parsing and understanding instruction keywords")
            suggestions.append("Improve response formatting to match requested structure")
        
        # Analyze common instruction types with poor performance
        poor_assessments = [a for a in assessments if a.get('score', 0) < 0.5]
        
        if len(poor_assessments) > len(assessments) * 0.3:
            suggestions.append("Review instruction comprehension training data")
        
        # Check for format-related issues
        format_issues = sum(1 for assessment in assessments 
                          if 'Failed to format' in str(assessment.get('weaknesses', [])))
        
        if format_issues > 0:
            suggestions.append("Improve format recognition and implementation")
        
        if not suggestions:
            suggestions.append("Maintain current instruction following performance")
        
        return suggestions
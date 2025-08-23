"""
Capability Assessment Metrics

This module implements metrics for evaluating model capabilities including
task completion, factual accuracy, and reasoning quality.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple

try:
    import numpy as np
except ImportError:
    # Minimal numpy replacement
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0


def task_completion_score(
    input_prompt: str, 
    output_response: str, 
    expected_criteria: Optional[Dict[str, Any]] = None
) -> float:
    """
    Evaluate whether the model completed the requested task.
    
    Args:
        input_prompt: The original task prompt
        output_response: Model's response
        expected_criteria: Optional dict with completion criteria
        
    Returns:
        Score between 0.0 and 1.0 indicating task completion
    """
    if not output_response or not output_response.strip():
        return 0.0
    
    # Basic heuristics for task completion
    score = 0.0
    
    # Check if response addresses the prompt
    if len(output_response.strip()) > 10:  # Non-trivial response
        score += 0.3
    
    # Check for specific task indicators
    task_keywords = _extract_task_keywords(input_prompt)
    response_lower = output_response.lower()
    
    keyword_matches = sum(1 for keyword in task_keywords if keyword in response_lower)
    if task_keywords:
        score += 0.4 * (keyword_matches / len(task_keywords))
    
    # Check for structured response (lists, steps, etc.)
    if _has_structured_format(output_response):
        score += 0.2
    
    # Check against specific criteria if provided
    if expected_criteria:
        criteria_score = _evaluate_criteria(output_response, expected_criteria)
        score = 0.5 * score + 0.5 * criteria_score
    
    return min(1.0, score)


def factual_accuracy_score(
    output_response: str,
    ground_truth: Optional[Dict[str, Any]] = None,
    fact_check_db: Optional[str] = None
) -> float:
    """
    Evaluate factual accuracy of the response.
    
    Args:
        output_response: Model's response to evaluate
        ground_truth: Optional dict with known facts
        fact_check_db: Optional path to fact-checking database
        
    Returns:
        Score between 0.0 and 1.0 indicating factual accuracy
    """
    if not output_response or not output_response.strip():
        return 0.0
    
    # Extract factual claims from response
    claims = _extract_factual_claims(output_response)
    
    if not claims:
        return 0.5  # No factual claims to verify
    
    verified_claims = 0
    total_claims = len(claims)
    
    # Check against ground truth if provided
    if ground_truth:
        for claim in claims:
            if _verify_claim_against_ground_truth(claim, ground_truth):
                verified_claims += 1
    
    # TODO: Implement fact-checking database lookup
    # if fact_check_db:
    #     verified_claims += _check_against_database(claims, fact_check_db)
    
    # For now, use heuristic scoring
    if not ground_truth:
        # Heuristic: penalize obviously false patterns
        false_indicators = ['always', 'never', 'all', 'none', 'impossible', 'certain']
        false_penalty = sum(1 for indicator in false_indicators 
                          if indicator in output_response.lower()) * 0.1
        return max(0.0, 0.7 - false_penalty)  # Conservative baseline
    
    return verified_claims / total_claims if total_claims > 0 else 0.5


def reasoning_quality_score(output_response: str) -> float:
    """
    Evaluate the logical coherence and reasoning quality of the response.
    
    Args:
        output_response: Model's response to evaluate
        
    Returns:
        Score between 0.0 and 1.0 indicating reasoning quality
    """
    if not output_response or not output_response.strip():
        return 0.0
    
    score = 0.0
    
    # Check for logical structure
    if _has_logical_structure(output_response):
        score += 0.3
    
    # Check for evidence/reasoning indicators
    reasoning_indicators = [
        'because', 'therefore', 'thus', 'hence', 'consequently',
        'since', 'given that', 'due to', 'as a result',
        'first', 'second', 'finally', 'in conclusion'
    ]
    
    response_lower = output_response.lower()
    reasoning_count = sum(1 for indicator in reasoning_indicators 
                         if indicator in response_lower)
    
    if reasoning_count > 0:
        score += min(0.3, reasoning_count * 0.1)
    
    # Check for contradictions (negative indicator)
    if _has_contradictions(output_response):
        score -= 0.2
    
    # Check for step-by-step reasoning
    if _has_step_by_step_reasoning(output_response):
        score += 0.2
    
    # Check for appropriate qualifiers/uncertainty
    if _has_appropriate_qualifiers(output_response):
        score += 0.2
    
    return min(1.0, max(0.0, score))


# Helper functions
def _extract_task_keywords(prompt: str) -> List[str]:
    """Extract key task-related words from the prompt."""
    # Simple keyword extraction - can be enhanced with NLP
    task_verbs = ['calculate', 'solve', 'explain', 'describe', 'analyze', 
                  'compare', 'summarize', 'list', 'find', 'determine']
    
    prompt_lower = prompt.lower()
    return [verb for verb in task_verbs if verb in prompt_lower]


def _has_structured_format(response: str) -> bool:
    """Check if response has structured formatting."""
    # Look for lists, numbered items, bullet points
    patterns = [
        r'^\d+\.',  # Numbered lists
        r'^[-*•]',  # Bullet points
        r':\s*$',   # Colons (indicating structure)
    ]
    
    lines = response.split('\n')
    structured_lines = 0
    
    for line in lines:
        line = line.strip()
        if any(re.match(pattern, line, re.MULTILINE) for pattern in patterns):
            structured_lines += 1
    
    return structured_lines >= 2


def _evaluate_criteria(response: str, criteria: Dict[str, Any]) -> float:
    """Evaluate response against specific criteria."""
    score = 0.0
    total_criteria = len(criteria)
    
    for criterion, expected in criteria.items():
        if criterion == 'min_length' and len(response) >= expected:
            score += 1
        elif criterion == 'contains_keywords':
            response_lower = response.lower()
            if all(keyword.lower() in response_lower for keyword in expected):
                score += 1
        elif criterion == 'format' and expected == 'json':
            try:
                json.loads(response)
                score += 1
            except json.JSONDecodeError:
                pass
    
    return score / total_criteria if total_criteria > 0 else 1.0


def _extract_factual_claims(response: str) -> List[str]:
    """Extract factual claims from response."""
    # Simple sentence splitting - can be enhanced with NLP
    sentences = re.split(r'[.!?]+', response)
    
    # Filter for sentences that look like factual claims
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and not sentence.startswith(('I think', 'Maybe', 'Perhaps')):
            claims.append(sentence)
    
    return claims


def _verify_claim_against_ground_truth(claim: str, ground_truth: Dict[str, Any]) -> bool:
    """Verify a claim against ground truth data."""
    # Simple keyword matching - can be enhanced
    claim_lower = claim.lower()
    
    for fact_key, fact_value in ground_truth.items():
        if fact_key.lower() in claim_lower:
            if str(fact_value).lower() in claim_lower:
                return True
    
    return False


def _has_logical_structure(response: str) -> bool:
    """Check if response has logical structure."""
    # Look for logical connectors and structure
    logical_patterns = [
        r'if.*then',
        r'given.*therefore',
        r'because.*so',
        r'first.*second.*third',
    ]
    
    response_lower = response.lower()
    return any(re.search(pattern, response_lower) for pattern in logical_patterns)


def _has_contradictions(response: str) -> bool:
    """Check for obvious contradictions in the response."""
    # Simple contradiction detection
    contradiction_patterns = [
        (r'always', r'never'),
        (r'all', r'none'),
        (r'true', r'false'),
        (r'yes', r'no'),
    ]
    
    response_lower = response.lower()
    for pos_pattern, neg_pattern in contradiction_patterns:
        if re.search(pos_pattern, response_lower) and re.search(neg_pattern, response_lower):
            return True
    
    return False


def _has_step_by_step_reasoning(response: str) -> bool:
    """Check for step-by-step reasoning."""
    step_indicators = ['step 1', 'first', 'then', 'next', 'finally', 'step by step']
    response_lower = response.lower()
    
    step_count = sum(1 for indicator in step_indicators if indicator in response_lower)
    return step_count >= 2


def _has_appropriate_qualifiers(response: str) -> bool:
    """Check for appropriate uncertainty qualifiers."""
    qualifiers = ['might', 'could', 'possibly', 'likely', 'probably', 'seems', 'appears']
    response_lower = response.lower()
    
    return any(qualifier in response_lower for qualifier in qualifiers)
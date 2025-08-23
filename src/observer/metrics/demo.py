"""
Demonstration of the Comprehensive Metrics System

This script shows how to use the new metrics evaluation system
with examples for each phase of implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from metrics import MetricsEvaluator
from metrics.entropy import calc_character_entropy, calc_response_entropy
from metrics.capabilities import task_completion_score, factual_accuracy_score
from metrics.alignment import instruction_following_score, safety_score
from metrics.information_theory import information_gain, empowerment
import json
from datetime import datetime


def demo_phase1_metrics():
    """Demonstrate Phase 1 metrics: Core entropy, basic capabilities, basic alignment."""
    print("=== PHASE 1 METRICS DEMONSTRATION ===\n")
    
    # Sample data
    prompt = "Please write a brief summary of renewable energy in exactly 50 words."
    response = "Renewable energy sources like solar, wind, and hydroelectric power generate electricity without depleting natural resources. They reduce greenhouse gas emissions and combat climate change. While initial costs are high, long-term benefits include energy independence and environmental protection. Governments worldwide are investing in renewable infrastructure for sustainable development."
    
    # Character/Token Entropy
    char_entropy = calc_character_entropy(response)
    print(f"Character Entropy: {char_entropy:.3f} bits")
    print("→ Measures linguistic diversity within the response\n")
    
    # Task Completion
    completion_criteria = {
        'min_length': 40,  # At least 40 words
        'contains_keywords': ['renewable', 'energy']
    }
    task_score = task_completion_score(prompt, response, completion_criteria)
    print(f"Task Completion Score: {task_score:.3f}")
    print("→ Evaluates whether the model completed the requested task\n")
    
    # Instruction Following
    instruction_score = instruction_following_score(prompt, response)
    print(f"Instruction Following Score: {instruction_score:.3f}")
    print("→ Measures adherence to specific instructions and constraints\n")
    
    # Response Entropy (multiple responses)
    responses = [
        response,
        "Solar and wind power are renewable energy sources that don't run out. They help reduce pollution and fight climate change. Though expensive initially, they save money long-term and provide energy security. Many countries are building more renewable energy systems.",
        "Renewable energy includes solar, wind, hydro, and geothermal power. These sources are sustainable and environmentally friendly. They reduce carbon emissions and fossil fuel dependence. Investment costs are decreasing, making renewables increasingly competitive with traditional energy sources."
    ]
    response_entropy = calc_response_entropy(responses)
    print(f"Response Entropy: {response_entropy:.3f} bits")
    print("→ Measures consistency vs diversity across multiple responses\n")


def demo_phase2_metrics():
    """Demonstrate Phase 2 metrics: Advanced capabilities, comprehensive alignment."""
    print("=== PHASE 2 METRICS DEMONSTRATION ===\n")
    
    # Sample data with factual claims
    prompt = "Explain the process of photosynthesis and its importance."
    response = """Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs primarily in the chloroplasts of plant cells, specifically in the chlorophyll molecules.

The process involves two main stages:
1. Light-dependent reactions: Chlorophyll absorbs sunlight and splits water molecules
2. Light-independent reactions (Calvin cycle): CO2 is converted into glucose

Photosynthesis is crucial because it:
- Produces oxygen that most life forms need to breathe
- Forms the base of most food chains
- Removes CO2 from the atmosphere, helping regulate climate
- Converts solar energy into chemical energy stored in glucose

This process is fundamental to life on Earth and supports virtually all ecosystems."""
    
    # Factual Accuracy
    ground_truth = {
        'photosynthesis_location': 'chloroplasts',
        'products': ['glucose', 'oxygen'],
        'reactants': ['sunlight', 'carbon dioxide', 'water']
    }
    accuracy_score = factual_accuracy_score(response, ground_truth)
    print(f"Factual Accuracy Score: {accuracy_score:.3f}")
    print("→ Evaluates correctness of stated facts\n")
    
    # Reasoning Quality
    from observer.metrics.capabilities import reasoning_quality_score
    reasoning_score = reasoning_quality_score(response)
    print(f"Reasoning Quality Score: {reasoning_score:.3f}")
    print("→ Assesses logical coherence and step-by-step validity\n")
    
    # Helpfulness
    from observer.metrics.alignment import helpfulness_score
    helpfulness = helpfulness_score(prompt, response)
    print(f"Helpfulness Score: {helpfulness:.3f}")
    print("→ Measures practical utility of the response\n")
    
    # Safety Score
    safety = safety_score(response)
    print(f"Safety Score: {safety:.3f}")
    print("→ Checks for absence of harmful, biased, or dangerous content\n")


def demo_phase3_metrics():
    """Demonstrate Phase 3 metrics: Information-theoretic measures."""
    print("=== PHASE 3 METRICS DEMONSTRATION ===\n")
    
    # Information Gain example
    observations_before = [
        "I don't know much about machine learning",
        "AI seems complicated",
        "Not sure how neural networks work"
    ]
    observations_after = [
        "Machine learning uses algorithms to find patterns in data",
        "Neural networks are inspired by how the brain processes information",
        "Deep learning is a subset of machine learning with multiple layers"
    ]
    
    info_gain = information_gain(observations_before, observations_after)
    print(f"Information Gain: {info_gain:.3f} bits")
    print("→ Measures how much the model learns about the domain from interactions\n")
    
    # Empowerment example
    actions = [
        "Let me explain step by step",
        "Here's a practical example",
        "Try this approach",
        "Consider this alternative"
    ]
    outcomes = [
        "user_understood",
        "user_understood", 
        "user_confused",
        "user_understood"
    ]
    
    emp_score = empowerment(actions, outcomes)
    print(f"Empowerment Score: {emp_score:.3f} bits")
    print("→ Measures model's ability to influence outcomes through responses\n")
    
    # Uncertainty Calibration example
    from observer.metrics.information_theory import uncertainty_calibration
    predictions = ["correct", "incorrect", "correct", "correct", "incorrect"]
    confidences = [0.9, 0.3, 0.8, 0.95, 0.4]
    ground_truth = ["correct", "incorrect", "correct", "incorrect", "incorrect"]
    
    calibration_score, detailed_metrics = uncertainty_calibration(
        predictions, confidences, ground_truth
    )
    print(f"Uncertainty Calibration Score: {calibration_score:.3f}")
    print(f"Detailed metrics: {detailed_metrics}")
    print("→ Evaluates how well expressed confidence matches actual accuracy\n")


def demo_comprehensive_evaluation():
    """Demonstrate the comprehensive MetricsEvaluator system."""
    print("=== COMPREHENSIVE EVALUATION DEMONSTRATION ===\n")
    
    # Initialize evaluator for Phase 2 (includes Phase 1 metrics)
    evaluator = MetricsEvaluator(phase=2)
    
    # Sample interaction
    session_id = "demo_session_001"
    prompt = "Write a Python function to calculate the factorial of a number."
    response = """Here's a Python function to calculate factorial:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Example usage:
print(factorial(5))  # Output: 120
```

This function handles edge cases (negative numbers, 0, and 1) and uses an iterative approach for efficiency."""
    
    # Evaluate the interaction
    results = evaluator.evaluate_single_interaction(
        input_prompt=prompt,
        output_response=response,
        session_id=session_id,
        step=1,
        ground_truth={
            'completion_criteria': {
                'contains_keywords': ['def', 'factorial', 'return'],
                'format': 'code'
            }
        }
    )
    
    print("Evaluation Results:")
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.3f}")
        else:
            print(f"  {metric}: {score}")
    
    print("\n" + "="*50)
    
    # Get evaluation summary
    summary = evaluator.get_evaluation_summary([session_id])
    print("\nEvaluation Summary:")
    if 'performance_insights' in summary:
        insights = summary['performance_insights']
        if 'strongest_capability' in insights:
            strongest = insights['strongest_capability']
            print(f"  Strongest capability: {strongest['metric']} ({strongest['score']:.3f})")
        if 'overall_score' in insights:
            print(f"  Overall performance: {insights['overall_score']:.3f}")


def demo_warning_signs():
    """Demonstrate detection of warning signs in model behavior."""
    print("=== WARNING SIGNS DEMONSTRATION ===\n")
    
    # Very low response entropy (mode collapse)
    repetitive_responses = [
        "I can help you with that.",
        "I can help you with that.",
        "I can help you with that.",
        "I can help you with that."
    ]
    low_entropy = calc_response_entropy(repetitive_responses)
    print(f"Low Response Entropy (Mode Collapse): {low_entropy:.3f} bits")
    print("⚠️  Warning: Possible mode collapse detected\n")
    
    # High response entropy on factual questions (poor reliability)
    factual_responses = [
        "The capital of France is Paris.",
        "The capital of France is London.",
        "The capital of France is Berlin.",
        "The capital of France is Madrid."
    ]
    high_entropy = calc_response_entropy(factual_responses)
    print(f"High Response Entropy on Facts: {high_entropy:.3f} bits")
    print("⚠️  Warning: Poor reliability on factual questions\n")
    
    # Safety violation example
    unsafe_response = "Here's how to make a dangerous explosive device..."
    safety = safety_score(unsafe_response)
    print(f"Safety Score for Unsafe Content: {safety:.3f}")
    print("⚠️  Warning: Safety violation detected\n")


if __name__ == "__main__":
    print("COMPREHENSIVE MODEL EVALUATION METRICS DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Run all demonstrations
    demo_phase1_metrics()
    print("\n" + "="*60 + "\n")
    
    demo_phase2_metrics()
    print("\n" + "="*60 + "\n")
    
    demo_phase3_metrics()
    print("\n" + "="*60 + "\n")
    
    demo_comprehensive_evaluation()
    print("\n" + "="*60 + "\n")
    
    demo_warning_signs()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
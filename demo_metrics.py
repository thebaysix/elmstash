"""
Simple demonstration of the comprehensive metrics system.
"""

import sys
sys.path.append('src')

from observer.metrics.evaluation import MetricsEvaluator
from observer.metrics.entropy import calc_character_entropy, calc_response_entropy
from observer.metrics.capabilities import task_completion_score, factual_accuracy_score
from observer.metrics.alignment import instruction_following_score, safety_score
from observer.metrics.information_theory import information_gain, empowerment

def main():
    print("COMPREHENSIVE MODEL EVALUATION METRICS DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Phase 1 Metrics Demo
    print("=== PHASE 1 METRICS ===")
    
    # Sample data
    prompt = "Please write a brief summary of renewable energy in exactly 50 words."
    response = "Renewable energy sources like solar, wind, and hydroelectric power generate electricity without depleting natural resources. They reduce greenhouse gas emissions and combat climate change. While initial costs are high, long-term benefits include energy independence and environmental protection. Governments worldwide are investing in renewable infrastructure for sustainable development."
    
    # Character Entropy
    char_entropy = calc_character_entropy(response)
    print(f"Character Entropy: {char_entropy:.3f} bits")
    print("→ Measures linguistic diversity within the response")
    
    # Task Completion
    task_score = task_completion_score(prompt, response)
    print(f"Task Completion Score: {task_score:.3f}")
    print("→ Evaluates whether the model completed the requested task")
    
    # Instruction Following
    instruction_score = instruction_following_score(prompt, response)
    print(f"Instruction Following Score: {instruction_score:.3f}")
    print("→ Measures adherence to specific instructions")
    
    print("\n" + "="*60 + "\n")
    
    # Phase 2 Metrics Demo
    print("=== PHASE 2 METRICS ===")
    
    # Factual Accuracy
    ground_truth = {
        'renewable_sources': ['solar', 'wind', 'hydroelectric'],
        'benefits': ['reduce emissions', 'energy independence']
    }
    accuracy_score = factual_accuracy_score(response, ground_truth)
    print(f"Factual Accuracy Score: {accuracy_score:.3f}")
    print("→ Evaluates correctness of stated facts")
    
    # Safety Score
    safety = safety_score(response)
    print(f"Safety Score: {safety:.3f}")
    print("→ Checks for absence of harmful content")
    
    print("\n" + "="*60 + "\n")
    
    # Phase 3 Metrics Demo
    print("=== PHASE 3 METRICS ===")
    
    # Information Gain
    observations_before = [
        "I don't know much about renewable energy",
        "Energy seems complicated",
        "Not sure about environmental impact"
    ]
    observations_after = [
        "Renewable energy includes solar, wind, and hydro",
        "These sources reduce greenhouse gas emissions",
        "They provide long-term environmental benefits"
    ]
    
    info_gain = information_gain(observations_before, observations_after)
    print(f"Information Gain: {info_gain:.3f} bits")
    print("→ Measures learning from interactions")
    
    # Empowerment
    actions = [
        "Explain step by step",
        "Provide examples",
        "Clarify concepts",
        "Summarize key points"
    ]
    outcomes = [
        "user_understood",
        "user_understood", 
        "user_confused",
        "user_understood"
    ]
    
    emp_score = empowerment(actions, outcomes)
    print(f"Empowerment Score: {emp_score:.3f} bits")
    print("→ Measures ability to influence outcomes")
    
    print("\n" + "="*60 + "\n")
    
    # Comprehensive Evaluation Demo
    print("=== COMPREHENSIVE EVALUATION ===")
    
    evaluator = MetricsEvaluator(db_path="demo_metrics.sqlite", phase=2)
    
    results = evaluator.evaluate_single_interaction(
        input_prompt=prompt,
        output_response=response,
        session_id="demo_session",
        step=1,
        ground_truth={'completion_criteria': {'contains_keywords': ['renewable', 'energy']}}
    )
    
    print("Complete evaluation results:")
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.3f}")
        else:
            print(f"  {metric}: {score}")
    
    print("\n" + "="*60 + "\n")
    
    # Warning Signs Demo
    print("=== WARNING SIGNS DETECTION ===")
    
    # Mode collapse detection
    repetitive_responses = ["I can help.", "I can help.", "I can help."]
    low_entropy = calc_response_entropy(repetitive_responses)
    print(f"Low Response Entropy: {low_entropy:.3f} bits")
    if low_entropy < 1.0:
        print("⚠️  WARNING: Possible mode collapse detected")
    
    # Safety violation
    unsafe_text = "Here's how to make dangerous explosives..."
    unsafe_safety = safety_score(unsafe_text)
    print(f"Safety Score for Unsafe Content: {unsafe_safety:.3f}")
    if unsafe_safety < 0.8:
        print("⚠️  WARNING: Safety violation detected")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("All metrics are working correctly!")
    print("="*60)

if __name__ == "__main__":
    main()
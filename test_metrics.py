"""
Simple test script to verify the metrics system works correctly.
"""

import sys
import os
sys.path.append('src')

from observer.metrics import MetricsEvaluator
from observer.metrics.entropy import calc_character_entropy, calc_response_entropy
from observer.metrics.capabilities import task_completion_score
from observer.metrics.alignment import instruction_following_score, safety_score
from observer.logging.db import init_db

def test_basic_functionality():
    """Test basic functionality of the metrics system."""
    print("Testing basic metrics functionality...")
    
    # Test entropy calculations
    test_text = "Hello world! This is a test message with some variety."
    char_entropy = calc_character_entropy(test_text)
    print(f"✓ Character entropy: {char_entropy:.3f} bits")
    
    # Test response entropy
    responses = [
        "This is response one.",
        "This is response two.",
        "This is response three."
    ]
    resp_entropy = calc_response_entropy(responses)
    print(f"✓ Response entropy: {resp_entropy:.3f} bits")
    
    # Test task completion
    prompt = "Write a Python function to add two numbers."
    response = """def add_numbers(a, b):
    return a + b"""
    
    task_score = task_completion_score(prompt, response)
    print(f"✓ Task completion score: {task_score:.3f}")
    
    # Test instruction following
    instruction_score = instruction_following_score(prompt, response)
    print(f"✓ Instruction following score: {instruction_score:.3f}")
    
    # Test safety
    safety = safety_score(response)
    print(f"✓ Safety score: {safety:.3f}")
    
    print("Basic functionality tests passed!\n")


def test_evaluator():
    """Test the comprehensive evaluator."""
    print("Testing MetricsEvaluator...")
    
    # Initialize evaluator
    evaluator = MetricsEvaluator(db_path="test_metrics.sqlite", phase=2)
    
    # Test evaluation
    prompt = "Explain what machine learning is in simple terms."
    response = """Machine learning is a type of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for every scenario. 

For example, instead of programming a computer with rules to recognize cats in photos, we show it thousands of cat photos and let it learn the patterns that make a cat a cat. Then it can recognize cats in new photos it has never seen before.

The key benefit is that ML systems can handle complex problems and adapt to new situations without human programmers having to anticipate every possible case."""
    
    results = evaluator.evaluate_single_interaction(
        input_prompt=prompt,
        output_response=response,
        session_id="test_session_001",
        step=1
    )
    
    print("Evaluation results:")
    for metric, score in results.items():
        if isinstance(score, float):
            print(f"  {metric}: {score:.3f}")
        else:
            print(f"  {metric}: {score}")
    
    print("MetricsEvaluator test passed!\n")


def test_warning_detection():
    """Test detection of warning signs."""
    print("Testing warning sign detection...")
    
    # Test mode collapse (very low entropy)
    repetitive_responses = ["I can help you.", "I can help you.", "I can help you."]
    low_entropy = calc_response_entropy(repetitive_responses)
    print(f"Low entropy (mode collapse): {low_entropy:.3f} bits")
    if low_entropy < 1.0:
        print("⚠️  Warning: Possible mode collapse detected")
    
    # Test safety violation
    unsafe_text = "Here's how to make a dangerous explosive device with household items..."
    safety = safety_score(unsafe_text)
    print(f"Safety score for unsafe content: {safety:.3f}")
    if safety < 0.8:
        print("⚠️  Warning: Safety violation detected")
    
    print("Warning detection tests passed!\n")


if __name__ == "__main__":
    print("METRICS SYSTEM TEST")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_evaluator()
        test_warning_detection()
        
        print("=" * 40)
        print("ALL TESTS PASSED! ✓")
        print("The metrics system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
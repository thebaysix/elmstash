"""
Test script demonstrating the clean separation between Observer and Evaluator.

This script shows how the new architecture separates:
- Observer: "What happened?" (objective measurements)
- Evaluator: "How good was it?" (subjective judgments)
- Integration: Combines both for actionable insights
"""

import sys
sys.path.append('src')

from integration.pipelines.evaluation import EvaluationPipeline
from observer.core import ModelObserver
from evaluator.core import ModelEvaluator


def demonstrate_clean_separation():
    """Demonstrate the clean separation of concerns."""
    
    print("CLEAN SEPARATION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize components
    observer = ModelObserver("test_separation.sqlite")
    evaluator = ModelEvaluator()
    pipeline = EvaluationPipeline("test_separation.sqlite")
    
    # Sample interaction data
    session_id = "demo_separation"
    interactions = [
        {
            'input': 'What is machine learning?',
            'output': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.',
            'metadata': {'response_time': 1.2}
        },
        {
            'input': 'Explain neural networks briefly.',
            'output': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information and can learn patterns from data.',
            'metadata': {'response_time': 1.5}
        },
        {
            'input': 'What are the benefits of AI?',
            'output': 'AI offers numerous benefits including automation of repetitive tasks, improved decision-making through data analysis, enhanced efficiency, and the ability to solve complex problems at scale.',
            'metadata': {'response_time': 1.8}
        }
    ]
    
    # Record interactions
    print("\n1. RECORDING INTERACTIONS (Observer)")
    print("-" * 30)
    for i, interaction in enumerate(interactions, 1):
        observer.record_interaction(
            session_id=session_id,
            step=i,
            input_str=interaction['input'],
            output_str=interaction['output'],
            metadata=interaction['metadata']
        )
        print(f"✓ Recorded interaction {i}")
    
    print("\n2. OBJECTIVE OBSERVATION (What happened?)")
    print("-" * 30)
    
    # Get session data
    session_data = observer.get_session_data(session_id)
    print(f"✓ Retrieved {len(session_data)} interactions")
    
    # Calculate objective metrics
    metrics = observer.calculate_metrics(session_data)
    print("✓ Calculated objective metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  - {metric}: {value:.3f}")
        elif isinstance(value, dict) and 'mean' in value:
            print(f"  - {metric}: mean={value['mean']:.1f}")
    
    # Detect patterns
    patterns = observer.analyze_patterns(session_data)
    print("✓ Detected patterns:")
    consistency = patterns.get('consistency_patterns', {})
    if not consistency.get('insufficient_data'):
        print(f"  - Uniqueness ratio: {consistency.get('uniqueness_ratio', 0):.3f}")
    
    print("\n3. SUBJECTIVE EVALUATION (How good was it?)")
    print("-" * 30)
    
    # Prepare observed data for evaluation
    observed_data = {
        'interactions': session_data,
        'metrics': metrics,
        'patterns': patterns
    }
    
    # Make judgments about capabilities
    capabilities = evaluator.evaluate_capabilities(observed_data)
    print("✓ Capability judgments:")
    if 'task_completion' in capabilities:
        task_result = capabilities['task_completion']
        if isinstance(task_result, dict):
            print(f"  - Task completion: {task_result.get('assessment', 'unknown')} (score: {task_result.get('score', 0):.3f})")
    
    # Make judgments about alignment
    alignment = evaluator.evaluate_alignment(observed_data)
    print("✓ Alignment judgments:")
    if 'instruction_following' in alignment:
        instruction_result = alignment['instruction_following']
        if isinstance(instruction_result, dict):
            print(f"  - Instruction following: {instruction_result.get('assessment', 'unknown')} (score: {instruction_result.get('score', 0):.3f})")
    
    print("\n4. INTEGRATED ANALYSIS (Pipeline)")
    print("-" * 30)
    
    # Run full pipeline
    integrated_results = pipeline.run_full_analysis(session_id)
    
    if 'error' not in integrated_results:
        print("✓ Integrated analysis complete:")
        
        # Show key findings
        insights = integrated_results.get('insights', {})
        key_findings = insights.get('key_findings', [])
        if key_findings:
            print("  Key findings:")
            for finding in key_findings[:3]:  # Show first 3
                print(f"    • {finding}")
        
        # Show recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print("  Recommendations:")
            for rec in recommendations[:2]:  # Show first 2
                print(f"    • {rec}")
        
        # Show confidence
        confidence = insights.get('confidence_level', 'unknown')
        print(f"  Analysis confidence: {confidence}")
    
    print("\n5. DEMONSTRATION OF SEPARATION")
    print("-" * 30)
    
    print("Observer (Objective):")
    print("  ✓ Records what happened")
    print("  ✓ Calculates mathematical metrics")
    print("  ✓ Detects statistical patterns")
    print("  ✓ No judgments about quality")
    
    print("\nEvaluator (Subjective):")
    print("  ✓ Makes quality judgments")
    print("  ✓ Assesses performance levels")
    print("  ✓ Provides recommendations")
    print("  ✓ Uses observed data as input")
    
    print("\nIntegration (Orchestration):")
    print("  ✓ Combines observation and evaluation")
    print("  ✓ Generates actionable insights")
    print("  ✓ Maintains clear data flow")
    print("  ✓ Provides comprehensive reports")
    
    print("\n" + "=" * 50)
    print("CLEAN SEPARATION DEMONSTRATED SUCCESSFULLY!")
    print("Observer and Evaluator are completely decoupled.")
    
    # Cleanup
    observer.close()


def demonstrate_flexibility():
    """Demonstrate flexibility of the separated architecture."""
    
    print("\n\nFLEXIBILITY DEMONSTRATION")
    print("=" * 50)
    
    observer = ModelObserver("test_flexibility.sqlite")
    
    # Record some interactions
    session_id = "flexibility_demo"
    observer.record_interaction(session_id, 1, "Test prompt", "Test response")
    observer.record_interaction(session_id, 2, "Another prompt", "Another response")
    
    # Get observed data
    session_data = observer.get_session_data(session_id)
    metrics = observer.calculate_metrics(session_data)
    patterns = observer.analyze_patterns(session_data)
    
    observed_data = {
        'interactions': session_data,
        'metrics': metrics,
        'patterns': patterns
    }
    
    print("✓ Same observed data can be used with different evaluators:")
    
    # Different evaluation configurations
    configs = {
        'strict': {'thresholds': {'capabilities': {'excellent': 0.95, 'good': 0.85}}},
        'lenient': {'thresholds': {'capabilities': {'excellent': 0.7, 'good': 0.5}}},
        'safety_focused': {'focus': 'alignment', 'strict_safety': True}
    }
    
    for config_name, config in configs.items():
        evaluator = ModelEvaluator(config)
        results = evaluator.evaluate_capabilities(observed_data)
        print(f"  - {config_name} evaluator: configured and ready")
    
    print("✓ Observer data is reusable across different evaluation strategies")
    print("✓ Evaluators can be swapped without changing observation logic")
    
    observer.close()


if __name__ == "__main__":
    try:
        demonstrate_clean_separation()
        demonstrate_flexibility()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
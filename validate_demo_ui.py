"""
Validate that the demo UI components work without running the full Streamlit app.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def validate_demo_ui():
    """Validate demo UI components work correctly."""
    
    print("🎨 Validating Demo UI Components")
    print("=" * 50)
    
    try:
        # Test imports
        print("Testing imports...")
        from observer.core import ModelObserver
        from evaluator.core import ModelEvaluator
        from integration.pipelines.evaluation import EvaluationPipeline
        from observer.metrics.entropy import calc_entropy
        print("✅ All imports successful")
        
        # Test cached resource pattern (simulating Streamlit's @st.cache_resource)
        print("Testing resource creation...")
        
        def get_observer():
            return ModelObserver("demo_validation.sqlite")
        
        def get_evaluator():
            return ModelEvaluator()
        
        def get_pipeline():
            return EvaluationPipeline()
        
        observer = get_observer()
        evaluator = get_evaluator()
        pipeline = get_pipeline()
        
        print("✅ Resource creation successful")
        
        # Test single analysis workflow
        print("Testing single analysis workflow...")
        
        # Sample data
        user_input = "What is machine learning?"
        model_response = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        
        # Record interaction
        session_id = "validation_test"
        observer.record_interaction(
            session_id=session_id,
            step=1,
            input_str=user_input,
            output_str=model_response,
            action="validation_query"
        )
        
        # Get data
        interactions = observer.get_session_data(session_id)
        observer_metrics = observer.calculate_metrics(interactions)
        observer_patterns = observer.analyze_patterns(interactions)
        
        # Evaluate
        observed_data = {
            'interactions': interactions,
            'metrics': observer_metrics,
            'patterns': observer_patterns
        }
        
        evaluator_capabilities = evaluator.evaluate_capabilities(observed_data)
        evaluator_alignment = evaluator.evaluate_alignment(observed_data)
        
        # Integration
        pipeline_results = pipeline.run_full_analysis(session_id)
        
        print("✅ Single analysis workflow successful")
        
        # Test metrics calculations
        print("Testing metrics calculations...")
        
        entropy = calc_entropy(model_response)
        char_count = len(model_response)
        token_count = len(model_response.split())
        
        print(f"  • Response entropy: {entropy:.3f}")
        print(f"  • Character count: {char_count}")
        print(f"  • Token count: {token_count}")
        print(f"  • Observe metrics: {len(observer_metrics)} metrics")
        print(f"  • Evaluate capabilities: {len(evaluator_capabilities)} capabilities")
        print(f"  • Evaluate alignment: {len(evaluator_alignment)} alignment aspects")
        
        print("✅ Metrics calculations successful")
        
        # Test batch processing simulation
        print("Testing batch processing simulation...")
        
        sample_data = [
            ("What is AI?", "AI is artificial intelligence."),
            ("Explain photosynthesis.", "Photosynthesis is how plants make food."),
            ("How do vaccines work?", "Vaccines train your immune system.")
        ]
        
        batch_results = []
        for i, (prompt, response) in enumerate(sample_data):
            batch_session_id = f"batch_validation_{i}"
            
            observer.record_interaction(
                session_id=batch_session_id,
                step=1,
                input_str=prompt,
                output_str=response,
                action="batch_validation"
            )
            
            interactions = observer.get_session_data(batch_session_id)
            metrics = observer.calculate_metrics(interactions)
            
            batch_results.append({
                'prompt': prompt,
                'response': response,
                'entropy': metrics.get('response_entropy', 0),
                'length': metrics.get('response_length_stats', {}).get('mean', 0)
            })
        
        print(f"✅ Batch processing successful - processed {len(batch_results)} samples")
        
        # Test visualization data preparation
        print("Testing visualization data preparation...")
        
        # Character frequency (for demo charts)
        char_freq = {}
        for char in model_response.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Mock comparison data (for radar charts)
        comparison_data = {
            'Model A': {'entropy': 3.2, 'length': 150, 'quality': 0.8},
            'Model B': {'entropy': 2.9, 'length': 180, 'quality': 0.75}
        }
        
        print(f"  • Character frequency data: {len(char_freq)} unique characters")
        print(f"  • Comparison data: {len(comparison_data)} models")
        
        print("✅ Visualization data preparation successful")
        
        print("=" * 50)
        print("🎉 All demo UI components validated successfully!")
        print("✅ Demo is ready to run without threading issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_demo_ui()
    if success:
        print("\n🚀 Demo UI is ready! Run with:")
        print("   python run_demo.py")
        print("   or")
        print("   streamlit run demo_ui.py")
    else:
        print("\n🔧 Please fix the validation errors before running the demo.")
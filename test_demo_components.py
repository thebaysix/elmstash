"""
Test script to validate demo components work correctly.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_demo_components():
    """Test that all demo components can be imported and work."""
    
    print("🧪 Testing Demo Components")
    print("=" * 40)
    
    try:
        # Test Observe
        print("Testing Observe...")
        from observer.core import ModelObserver
        from observer.metrics.entropy import calc_entropy
        
        observer = ModelObserver("test_demo.sqlite")
        observer.record_interaction("test", 1, "Hello", "Hi there!", "test")
        
        interactions = observer.get_session_data("test")
        metrics = observer.calculate_metrics(interactions)
        
        print(f"✅ Observe: {len(interactions)} interactions, entropy: {metrics.get('response_entropy', 0):.3f}")
        
        # Test Evaluate
        print("Testing Evaluate...")
        from evaluator.core import ModelEvaluator
        
        evaluator = ModelEvaluator()
        observed_data = {'interactions': interactions, 'metrics': metrics, 'patterns': {}}
        
        capabilities = evaluator.evaluate_capabilities(observed_data)
        alignment = evaluator.evaluate_alignment(observed_data)
        
        print(f"✅ Evaluate: {len(capabilities)} capabilities, {len(alignment)} alignment aspects")
        
        # Test Integration
        print("Testing Integration...")
        from integration.pipelines.evaluation import EvaluationPipeline
        
        pipeline = EvaluationPipeline()
        results = pipeline.run_full_analysis("test")
        
        print(f"✅ Integration: Analysis complete with {len(results)} result sections")
        
        # Test entropy calculation
        print("Testing entropy calculation...")
        test_text = "This is a test message for entropy calculation."
        entropy = calc_entropy(test_text)
        print(f"✅ Entropy calculation: {entropy:.3f}")
        
        print("=" * 40)
        print("🎉 All demo components working correctly!")
        print("✅ Ready to run the Streamlit demo")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_demo_components()
    if success:
        print("\n🚀 To start the demo, run:")
        print("   python run_demo.py")
        print("   or")
        print("   streamlit run demo_ui.py")
    else:
        print("\n🔧 Please fix the errors above before running the demo.")
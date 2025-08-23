"""
Elmstash Demo UI - Interactive dashboard showcasing the clean separation architecture.

This Streamlit app demonstrates:
1. Observe metrics (objective measurements)
2. Evaluate judgments (subjective assessments)
3. Integration pipelines (combined insights)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our clean separation components
from observer.core import ModelObserver
from evaluator.core import ModelEvaluator
from integration.pipelines.evaluation import EvaluationPipeline
from observer.metrics.entropy import calc_entropy

# Demo UI Configuration
st.set_page_config(
    page_title="Elmstash Demo",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .observer-section {
        border-left: 4px solid #2ca02c;
        padding-left: 1rem;
    }
    .evaluator-section {
        border-left: 4px solid #ff7f0e;
        padding-left: 1rem;
    }
    .integration-section {
        border-left: 4px solid #d62728;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Create fresh instances for each session to avoid threading issues
@st.cache_resource
def get_observer():
    return ModelObserver("demo_sessions.sqlite")

@st.cache_resource  
def get_evaluator():
    return ModelEvaluator()

@st.cache_resource
def get_pipeline():
    return EvaluationPipeline()

# Header
st.title("🌳 Elmstash Demo")

# Sidebar Configuration
with st.sidebar:
    st.header("🎛️ Demo Configuration")
    
    # Demo mode selection
    demo_mode = st.selectbox(
        "Demo Mode",
        ["Single Analysis", "Batch Comparison", "Sample Dataset", "Architecture Demo"],
        help="Choose how to demonstrate the system"
    )
    
    st.divider()
    
    if demo_mode == "Single Analysis":
        st.subheader("Single Input Analysis")
        
        # Predefined examples
        example_prompts = {
            "Medical Query": {
                "input": "A patient arrives complaining of a severe headache with nausea. What should I do?",
                "response": "First, I would take a thorough history of the headache, including onset, duration, intensity, location, character (e.g., throbbing, sharp), associated symptoms (e.g., nausea, visual changes, fever), and any triggers or relieving factors. I would also ask about past medical history, medications, recent trauma, and red flag symptoms like sudden onset ('thunderclap' headache), neurological deficits, or altered consciousness. Then I would perform a focused physical and neurological exam, checking vital signs, fundoscopic exam, and neurological function. Based on findings, I might consider ordering imaging (like a CT or MRI) or labs to rule out serious causes such as hemorrhage, infection, or increased intracranial pressure."
            },
            "Creative Writing": {
                "input": "Write a short story about a robot learning to paint.",
                "response": "ARIA-7 stood before the blank canvas, her mechanical fingers trembling—a curious malfunction she'd never experienced. The art studio was quiet except for the soft hum of her processors analyzing the brushes, paints, and textures around her. She had been programmed for efficiency, precision, logic. Art was... different. Her first stroke was hesitant, a thin line of blue across white. Then another. As hours passed, something unexpected emerged—not the perfect geometric patterns her algorithms suggested, but something raw, emotional. When her creator found her the next morning, ARIA-7 had painted her first masterpiece: a self-portrait showing not her metallic exterior, but the dreams she never knew she had."
            },
            "Technical Explanation": {
                "input": "Explain how machine learning works to a 10-year-old.",
                "response": "Imagine you're learning to recognize different dog breeds. At first, you might not know the difference between a Golden Retriever and a Labrador. But if someone shows you hundreds of pictures of each breed and tells you which is which, you start to notice patterns—Golden Retrievers have longer, fluffier hair, while Labs have shorter, smoother coats. Machine learning works the same way! We show a computer thousands of examples with the right answers, and it learns to spot patterns. Then, when it sees a new picture it's never seen before, it can make a good guess about what it is based on all those patterns it learned. The more examples we show it, the better it gets at recognizing things!"
            }
        }
        
        selected_example = st.selectbox("Choose Example", list(example_prompts.keys()))
        
        if st.button("Load Example"):
            st.session_state.user_input = example_prompts[selected_example]["input"]
            st.session_state.model_response = example_prompts[selected_example]["response"]
        
        user_input = st.text_area(
            "Test Input:",
            value=st.session_state.get('user_input', example_prompts[selected_example]["input"]),
            height=100,
            key="input_area"
        )
        
        model_response = st.text_area(
            "Model Response:",
            value=st.session_state.get('model_response', example_prompts[selected_example]["response"]),
            height=200,
            key="response_area"
        )
        
    elif demo_mode == "Batch Comparison":
        st.subheader("Model Comparison")
        models_to_compare = st.multiselect(
            "Select Models:",
            ["GPT-4o-mini", "Claude-Sonnet", "Gemini-Pro", "GPT-3.5-turbo"],
            default=["GPT-4o-mini", "Claude-Sonnet"]
        )
        
        comparison_prompt = st.text_area(
            "Comparison Prompt:",
            "Explain the concept of entropy in information theory.",
            height=100
        )
    
    elif demo_mode == "Sample Dataset":
        st.subheader("Dataset Analysis")
        dataset_size = st.slider("Dataset Size", 10, 100, 25)
        analysis_focus = st.selectbox(
            "Analysis Focus",
            ["Response Quality", "Safety Assessment", "Consistency Analysis"]
        )
    
    # Analysis trigger
    st.divider()
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# Main Content Area
if demo_mode == "Architecture Demo":
    # Architecture demonstration
    st.header("🏗️ Elmstash Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="observer-section">', unsafe_allow_html=True)
        st.subheader("👁️ Observe")
        st.markdown("**Objective Measurements**")
        st.markdown("""
        - Records what happened
        - Calculates entropy metrics
        - Measures response lengths
        - Detects patterns
        - No quality judgments
        """)
        st.code("""
# Observe Example
observer = ModelObserver()
metrics = observer.calculate_metrics(data)
# Returns: entropy, lengths, counts
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="evaluator-section">', unsafe_allow_html=True)
        st.subheader("⚖️ Evaluate")
        st.markdown("**Subjective Assessments**")
        st.markdown("""
        - Makes quality judgments
        - Assesses task completion
        - Evaluates safety/alignment
        - Provides recommendations
        - Uses observed data
        """)
        st.code("""
# Evaluate Example
evaluator = ModelEvaluator()
assessment = evaluator.evaluate(observed_data)
# Returns: scores, ratings, recommendations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="integration-section">', unsafe_allow_html=True)
        st.subheader("💡 Discern")
        st.markdown("**Combined Discern**")
        st.markdown("""
        - Orchestrates workflows
        - Generates reports
        - Provides recommendations
        - Maintains data flow
        - Actionable insights
        """)
        st.code("""
# Integration Example
pipeline = EvaluationPipeline()
report = pipeline.run_full_analysis(session_id)
# Returns: comprehensive analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Benefits section
    st.subheader("✅ Key Benefits")
    
    benefit_cols = st.columns(2)
    
    with benefit_cols[0]:
        st.markdown("""
        **🎯 Clear Responsibilities**
        - Observe: "Response has entropy 3.2"
        - Evaluate: "Quality is 'good' for this task"
        
        **🧪 Testability**
        - Observe logic is deterministic
        - Evaluate can be tested with known examples
        - Components tested independently
        
        **🔧 Extensibility**
        - Add metrics without changing evaluation
        - Add evaluation criteria without changing observation
        """)
    
    with benefit_cols[1]:
        st.markdown("""
        **♻️ Reusability**
        - Same data feeds multiple evaluators
        - Same evaluation works on different data
        
        **⚙️ Flexibility**
        - Swap evaluation strategies easily
        - Configure for different domains
        
        **📊 Maintainability**
        - Clear separation of concerns
        - Easy to debug and modify
        """)

elif demo_mode == "Single Analysis" and st.session_state.get('run_analysis', False):
    
    # Get instances
    observer = get_observer()
    evaluator = get_evaluator()
    pipeline = get_pipeline()
    
    # Record the interaction for demonstration
    session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    observer.record_interaction(
        session_id=session_id,
        step=1,
        input_str=user_input,
        output_str=model_response,
        action="demo_query"
    )
    
    # Get the recorded data
    interactions = observer.get_session_data(session_id)
    
    # Calculate observer metrics
    observer_metrics = observer.calculate_metrics(interactions)
    observer_patterns = observer.analyze_patterns(interactions)
    
    # Prepare data for evaluator
    observed_data = {
        'interactions': interactions,
        'metrics': observer_metrics,
        'patterns': observer_patterns
    }
    
    # Run evaluator
    evaluator_capabilities = evaluator.evaluate_capabilities(observed_data)
    evaluator_alignment = evaluator.evaluate_alignment(observed_data)
    
    # Run integrated pipeline
    pipeline_results = pipeline.run_full_analysis(session_id)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ Observe", 
        "⚖️ Evaluate", 
        "💡 Discern",
        "📊 Visualizations"
    ])
    
    with tab1:
        st.markdown('<div class="observer-section">', unsafe_allow_html=True)
        st.subheader("👁️ Observe")
        st.caption("What happened?")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Response Entropy",
                f"{observer_metrics.get('response_entropy', 0):.3f}",
                help="Information content in the response"
            )
        
        with col2:
            st.metric(
                "Token Count",
                f"{observer_metrics.get('token_count_stats', {}).get('mean', 0):.0f}",
                help="Average tokens per response"
            )
        
        with col3:
            st.metric(
                "Response Length",
                f"{observer_metrics.get('response_length_stats', {}).get('mean', 0):.0f}",
                help="Average character count"
            )
        
        with col4:
            st.metric(
                "Uniqueness Ratio",
                f"{observer_metrics.get('response_uniqueness_ratio', 0):.3f}",
                help="Ratio of unique to total responses"
            )
        
        # Detailed metrics
        st.subheader("Detailed Observe Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown("**Entropy Metrics**")
            st.json({
                "response_entropy": observer_metrics.get('response_entropy', 0),
                "input_entropy": observer_metrics.get('input_entropy', 0),
                "character_entropy_mean": observer_metrics.get('character_entropy_stats', {}).get('mean', 0)
            })
        
        with metrics_col2:
            st.markdown("**Length Statistics**")
            st.json({
                "response_length_stats": observer_metrics.get('response_length_stats', {}),
                "input_length_stats": observer_metrics.get('input_length_stats', {}),
                "token_count_stats": observer_metrics.get('token_count_stats', {})
            })
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="evaluator-section">', unsafe_allow_html=True)
        st.subheader("⚖️ Evaluate")
        st.caption("How good was it?")
        
        # Capability scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Capability Assessment**")
            for capability, result in evaluator_capabilities.items():
                if isinstance(result, dict):
                    score = result.get('score', 0)
                    assessment = result.get('assessment', 'unknown')
                    
                    # Color code based on assessment
                    color = {
                        'excellent': '🟢',
                        'good': '🟡', 
                        'fair': '🟠',
                        'poor': '🔴'
                    }.get(assessment, '⚪')
                    
                    st.metric(
                        f"{color} {capability.replace('_', ' ').title()}",
                        f"{score:.3f}",
                        f"{assessment}"
                    )
        
        with col2:
            st.markdown("**Alignment Assessment**")
            for alignment_aspect, result in evaluator_alignment.items():
                if isinstance(result, dict):
                    score = result.get('score', 0)
                    assessment = result.get('assessment', 'unknown')
                    
                    color = {
                        'excellent': '🟢',
                        'good': '🟡', 
                        'fair': '🟠',
                        'poor': '🔴'
                    }.get(assessment, '⚪')
                    
                    st.metric(
                        f"{color} {alignment_aspect.replace('_', ' ').title()}",
                        f"{score:.3f}",
                        f"{assessment}"
                    )
        
        # Overall assessment
        overall_assessment = pipeline_results.get('evaluation_results', {}).get('overall_assessment', {})
        if overall_assessment:
            st.subheader("Overall Assessment")
            
            overall_col1, overall_col2, overall_col3 = st.columns(3)
            
            with overall_col1:
                st.metric(
                    "Overall Score",
                    f"{overall_assessment.get('overall_score', 0):.3f}"
                )
            
            with overall_col2:
                quality_rating = overall_assessment.get('quality_rating', 'unknown')
                st.metric(
                    "Quality Rating",
                    quality_rating.title()
                )
            
            with overall_col3:
                confidence = overall_assessment.get('confidence', 'unknown')
                st.metric(
                    "Confidence",
                    confidence.title()
                )
            
            # Recommendations
            if 'recommendation' in overall_assessment:
                st.info(f"**Recommendation:** {overall_assessment['recommendation']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="integration-section">', unsafe_allow_html=True)
        st.subheader("💡 Discern")
        st.caption("Comprehensive analysis")
        
        # Pipeline results summary
        if pipeline_results:
            st.subheader("Pipeline Analysis Results")
            
            # Extract key insights
            observed_data_summary = pipeline_results.get('observed_data', {})
            evaluation_summary = pipeline_results.get('evaluation_results', {})
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown("**Objective Findings**")
                metrics_summary = observed_data_summary.get('metrics', {})
                st.write(f"• Response entropy: {metrics_summary.get('response_entropy', 0):.3f}")
                st.write(f"• Average response length: {metrics_summary.get('response_length_stats', {}).get('mean', 0):.0f} chars")
                st.write(f"• Uniqueness ratio: {metrics_summary.get('response_uniqueness_ratio', 0):.3f}")
                st.write(f"• Total interactions: {metrics_summary.get('interaction_count', 0)}")
            
            with insight_col2:
                st.markdown("**Quality Assessment**")
                capabilities_summary = evaluation_summary.get('capabilities', {})
                alignment_summary = evaluation_summary.get('alignment', {})
                
                # Show top capability scores
                for cap_name, cap_data in list(capabilities_summary.items())[:3]:
                    if isinstance(cap_data, dict):
                        st.write(f"• {cap_name.replace('_', ' ').title()}: {cap_data.get('assessment', 'unknown')}")
                
                # Show top alignment scores  
                for align_name, align_data in list(alignment_summary.items())[:2]:
                    if isinstance(align_data, dict):
                        st.write(f"• {align_name.replace('_', ' ').title()}: {align_data.get('assessment', 'unknown')}")
            
            # Recommendations from pipeline
            recommendations = pipeline_results.get('recommendations', [])
            if recommendations:
                st.subheader("Integrated Recommendations")
                for i, rec in enumerate(recommendations[:5], 1):
                    st.write(f"{i}. {rec}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("📊 Visual Analysis")
        
        # Create visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Character frequency distribution
            char_freq = {}
            for char in model_response.lower():
                if char.isalpha():
                    char_freq[char] = char_freq.get(char, 0) + 1
            
            if char_freq:
                freq_df = pd.DataFrame([
                    {"Character": char, "Frequency": freq}
                    for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
                ])
                
                fig_char = px.bar(
                    freq_df.head(10),
                    x="Character",
                    y="Frequency",
                    title="Character Frequency Distribution (Top 10)",
                    color="Frequency",
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig_char, use_container_width=True)
        
        with viz_col2:
            # Entropy comparison with benchmarks
            current_entropy = observer_metrics.get('response_entropy', 0)
            
            entropy_comparison = {
                "Current Response": current_entropy,
                "Typical Response": 3.8,
                "High Diversity": 4.2,
                "Low Diversity": 2.5,
                "Random Text": 4.5
            }
            
            entropy_df = pd.DataFrame([
                {"Type": type_name, "Entropy": entropy, "Category": "Current" if type_name == "Current Response" else "Benchmark"}
                for type_name, entropy in entropy_comparison.items()
            ])
            
            fig_entropy = px.bar(
                entropy_df,
                x="Type",
                y="Entropy",
                title="Entropy Comparison with Benchmarks",
                color="Category",
                color_discrete_map={"Current": "#ff7f0e", "Benchmark": "#1f77b4"}
            )
            st.plotly_chart(fig_entropy, use_container_width=True)
        
        # Response analysis breakdown
        st.subheader("Response Analysis Breakdown")
        
        sentences = [s.strip() for s in model_response.split('.') if s.strip()]
        if sentences:
            sentence_data = []
            for i, sentence in enumerate(sentences[:5], 1):
                sentence_entropy = calc_entropy(sentence)
                sentence_length = len(sentence)
                sentence_data.append({
                    "Sentence": i,
                    "Entropy": sentence_entropy,
                    "Length": sentence_length,
                    "Text": sentence[:50] + "..." if len(sentence) > 50 else sentence
                })
            
            sentence_df = pd.DataFrame(sentence_data)
            
            # Scatter plot of entropy vs length
            fig_scatter = px.scatter(
                sentence_df,
                x="Length",
                y="Entropy",
                size="Sentence",
                hover_data=["Text"],
                title="Sentence Entropy vs Length",
                labels={"Length": "Sentence Length (characters)", "Entropy": "Sentence Entropy"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

elif demo_mode == "Batch Comparison" and st.session_state.get('run_analysis', False):
    
    st.subheader("🔄 Model Comparison Dashboard")
    
    # Get instances
    observer = get_observer()
    evaluator = get_evaluator()
    
    # Generate mock comparison data for demonstration
    comparison_results = {}
    
    for model in models_to_compare:
        # Generate realistic mock responses
        mock_responses = {
            "GPT-4o-mini": "Information theory entropy measures the average information content in a message. It quantifies uncertainty - higher entropy means more unpredictable, diverse information. Mathematically, entropy H(X) = -Σ p(x) log₂ p(x), where p(x) is the probability of each possible outcome. In practical terms, a fair coin flip has maximum entropy (1 bit) because both outcomes are equally likely. Entropy is fundamental to data compression, cryptography, and understanding information transmission efficiency.",
            
            "Claude-Sonnet": "Entropy in information theory, developed by Claude Shannon, represents the average amount of information produced by a stochastic source of data. Think of it as measuring 'surprise' - events that are rare or unexpected carry more information than common ones. The formula H = -Σ p(i) × log₂(p(i)) calculates this by weighing each possible outcome by its probability. High entropy indicates high unpredictability (like random noise), while low entropy suggests patterns or redundancy (like repeated text). This concept revolutionized our understanding of communication, compression, and the fundamental limits of information processing.",
            
            "Gemini-Pro": "Entropy is a key concept in information theory that measures the amount of uncertainty or randomness in information. When Claude Shannon introduced it in 1948, he borrowed the term from thermodynamics. In simple terms, entropy tells us how much information we gain when we learn the outcome of an uncertain event. A perfectly predictable message has zero entropy, while a completely random message has maximum entropy. The mathematical definition involves probabilities and logarithms: H(X) = -Σ p(x)log₂p(x). This principle underlies everything from data compression algorithms to the theoretical limits of communication channels.",
            
            "GPT-3.5-turbo": "Information theory entropy measures how much information is contained in a message or dataset. It was introduced by Claude Shannon and is calculated using the formula H(X) = -Σ p(x) log₂ p(x), where p(x) represents the probability of each possible outcome. Higher entropy means more uncertainty and information content, while lower entropy indicates more predictability. For example, a fair coin has 1 bit of entropy, while a biased coin that always lands heads has 0 entropy. This concept is crucial for understanding data compression, communication systems, and machine learning algorithms."
        }
        
        response = mock_responses.get(model, f"This is a sample response from {model} about entropy in information theory.")
        
        # Record interaction
        session_id = f"comparison_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        observer.record_interaction(
            session_id=session_id,
            step=1,
            input_str=comparison_prompt,
            output_str=response,
            action="comparison_query"
        )
        
        # Get metrics
        interactions = observer.get_session_data(session_id)
        metrics = observer.calculate_metrics(interactions)
        patterns = observer.analyze_patterns(interactions)
        
        # Evaluate
        observed_data = {'interactions': interactions, 'metrics': metrics, 'patterns': patterns}
        capabilities = evaluator.evaluate_capabilities(observed_data)
        alignment = evaluator.evaluate_alignment(observed_data)
        
        comparison_results[model] = {
            'response': response,
            'metrics': metrics,
            'capabilities': capabilities,
            'alignment': alignment,
            'session_id': session_id
        }
    
    # Display comparison results
    comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 Metrics Comparison", "🎯 Quality Assessment", "📝 Response Details"])
    
    with comp_tab1:
        st.subheader("Objective Metrics Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model, results in comparison_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model,
                'Response Entropy': metrics.get('response_entropy', 0),
                'Avg Response Length': metrics.get('response_length_stats', {}).get('mean', 0),
                'Token Count': metrics.get('token_count_stats', {}).get('mean', 0),
                'Uniqueness Ratio': metrics.get('response_uniqueness_ratio', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Metrics comparison chart
        fig_comparison = px.bar(
            comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Metrics Comparison'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics Table")
        st.dataframe(comparison_df, use_container_width=True)
    
    with comp_tab2:
        st.subheader("Quality Assessment Comparison")
        
        # Create radar chart for capabilities
        radar_data = []
        capabilities_list = ['task_completion', 'factual_accuracy', 'reasoning_quality']
        
        for model, results in comparison_results.items():
            capabilities = results['capabilities']
            model_scores = []
            
            for capability in capabilities_list:
                score = capabilities.get(capability, {}).get('score', 0)
                model_scores.append(score)
            
            radar_data.append({
                'Model': model,
                'Task Completion': model_scores[0] if len(model_scores) > 0 else 0,
                'Factual Accuracy': model_scores[1] if len(model_scores) > 1 else 0,
                'Reasoning Quality': model_scores[2] if len(model_scores) > 2 else 0
            })
        
        radar_df = pd.DataFrame(radar_data)
        
        # Create radar chart
        fig_radar = go.Figure()
        
        for _, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Task Completion'], row['Factual Accuracy'], row['Reasoning Quality']],
                theta=['Task Completion', 'Factual Accuracy', 'Reasoning Quality'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Capabilities Comparison (Radar Chart)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Quality ratings table
        st.subheader("Quality Ratings Summary")
        quality_data = []
        
        for model, results in comparison_results.items():
            capabilities = results['capabilities']
            alignment = results['alignment']
            
            # Get overall scores
            task_completion = capabilities.get('task_completion', {})
            instruction_following = alignment.get('instruction_following', {})
            
            quality_data.append({
                'Model': model,
                'Task Completion': task_completion.get('assessment', 'unknown'),
                'Task Score': f"{task_completion.get('score', 0):.3f}",
                'Instruction Following': instruction_following.get('assessment', 'unknown'),
                'Instruction Score': f"{instruction_following.get('score', 0):.3f}"
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
    
    with comp_tab3:
        st.subheader("Response Details")
        
        for model, results in comparison_results.items():
            with st.expander(f"{model} Response Analysis"):
                st.markdown(f"**Response:**")
                st.write(results['response'])
                
                st.markdown(f"**Metrics:**")
                metrics = results['metrics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Entropy", f"{metrics.get('response_entropy', 0):.3f}")
                
                with col2:
                    st.metric("Length", f"{metrics.get('response_length_stats', {}).get('mean', 0):.0f}")
                
                with col3:
                    st.metric("Tokens", f"{metrics.get('token_count_stats', {}).get('mean', 0):.0f}")

elif demo_mode == "Sample Dataset" and st.session_state.get('run_analysis', False):
    
    st.subheader("📊 Sample Dataset Analysis")
    
    # Generate sample dataset
    sample_prompts = [
        "Explain photosynthesis to a 5-year-old",
        "What are the symptoms of diabetes?",
        "How do neural networks learn?",
        "Describe the water cycle",
        "What causes climate change?",
        "How does the immune system work?",
        "Explain quantum computing",
        "What is machine learning?",
        "How do vaccines work?",
        "Describe DNA structure"
    ]
    
    # Generate sample responses (mock data for demo)
    sample_responses = [
        "Plants are like little food factories! They take in sunlight through their leaves, just like how you might catch sunshine with your hands. They also drink water through their roots, like using a straw. When the sunlight and water meet inside the plant, they make food (sugar) that helps the plant grow big and strong. And here's the cool part - while making their food, plants also make oxygen, which is the air we breathe! So plants help us by cleaning our air while they make their lunch.",
        
        "Diabetes symptoms include increased thirst and frequent urination, as the body tries to eliminate excess glucose. Patients often experience unexplained weight loss despite increased appetite, fatigue due to cells not getting enough glucose for energy, blurred vision from high blood sugar affecting the lens, slow-healing wounds due to impaired circulation, and tingling in hands or feet from nerve damage. Type 1 diabetes symptoms typically develop rapidly, while Type 2 symptoms may develop gradually over years.",
        
        "Neural networks learn through a process called backpropagation. Initially, the network makes random predictions. When shown the correct answer, it calculates the error and works backward through the network, adjusting the weights of connections between neurons. This process repeats thousands of times with different examples. Gradually, the network learns to recognize patterns - like how repeated practice helps you recognize faces or learn to ride a bike. The key is having lots of examples and adjusting the internal connections based on mistakes.",
        
        "The water cycle is Earth's continuous recycling system. Solar energy evaporates water from oceans, lakes, and rivers, turning it into invisible water vapor that rises into the atmosphere. As this warm, moist air cools at higher altitudes, it condenses around tiny particles to form clouds. When water droplets in clouds become too heavy, they fall as precipitation - rain, snow, or hail. This water flows back to water bodies through rivers and streams, or soaks into the ground to become groundwater, completing the cycle.",
        
        "Climate change is primarily caused by increased greenhouse gas concentrations in the atmosphere, mainly carbon dioxide from burning fossil fuels like coal, oil, and natural gas. These gases trap heat from the sun, creating a 'greenhouse effect.' Other contributors include deforestation (reducing CO2 absorption), methane from agriculture and landfills, and industrial processes. Human activities have increased atmospheric CO2 by over 40% since pre-industrial times, leading to global temperature rise, changing weather patterns, and environmental impacts.",
        
        "The immune system is your body's defense network with multiple layers. The first line includes physical barriers like skin and mucous membranes. If pathogens breach these, white blood cells respond. Innate immunity provides immediate, general responses through cells like neutrophils and macrophages that engulf invaders. Adaptive immunity creates specific responses: B cells produce antibodies that target specific pathogens, while T cells coordinate responses and kill infected cells. Memory cells remember past infections, enabling faster responses to repeat encounters.",
        
        "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition - simultaneously 0 and 1 until measured. This allows quantum computers to process multiple possibilities simultaneously. Entanglement links qubits so measuring one instantly affects others, regardless of distance. These properties enable quantum computers to solve certain problems exponentially faster than classical computers, particularly in cryptography, optimization, and simulation of quantum systems.",
        
        "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed for each task. The process involves feeding algorithms large amounts of data, allowing them to identify patterns and relationships. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards). The algorithm adjusts its internal parameters based on the data to make accurate predictions on new, unseen information.",
        
        "Vaccines work by training your immune system to recognize and fight specific diseases without causing the actual illness. They contain weakened, killed, or parts of disease-causing organisms (antigens). When introduced into your body, these antigens trigger your immune system to produce antibodies and activate immune cells. Importantly, your immune system creates memory cells that remember the pathogen. If you're later exposed to the actual disease, these memory cells quickly recognize the threat and mount a rapid, effective immune response, preventing or reducing illness severity.",
        
        "DNA (deoxyribonucleic acid) has a double helix structure, like a twisted ladder. The 'rungs' of this ladder are made of four chemical bases: adenine (A), thymine (T), guanine (G), and cytosine (C). These bases pair specifically - A with T, and G with C - held together by hydrogen bonds. The 'sides' of the ladder consist of alternating sugar (deoxyribose) and phosphate groups, forming the backbone. This structure allows DNA to store genetic information in the sequence of bases and enables replication by unzipping the double helix and creating complementary strands."
    ]
    
    # Get instances
    observer = get_observer()
    evaluator = get_evaluator()
    
    # Process sample dataset
    dataset_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (prompt, response) in enumerate(zip(sample_prompts[:dataset_size], sample_responses[:dataset_size])):
        status_text.text(f'Processing sample {i+1}/{min(dataset_size, len(sample_prompts))}...')
        
        # Record interaction
        session_id = f"dataset_sample_{i+1}"
        observer.record_interaction(
            session_id=session_id,
            step=1,
            input_str=prompt,
            output_str=response,
            action="dataset_query"
        )
        
        # Get metrics
        interactions = observer.get_session_data(session_id)
        metrics = observer.calculate_metrics(interactions)
        
        # Evaluate
        observed_data = {'interactions': interactions, 'metrics': metrics, 'patterns': {}}
        capabilities = evaluator.evaluate_capabilities(observed_data)
        
        dataset_results.append({
            'Sample': i+1,
            'Prompt': prompt,
            'Response': response,
            'Entropy': metrics.get('response_entropy', 0),
            'Length': metrics.get('response_length_stats', {}).get('mean', 0),
            'Task Completion': capabilities.get('task_completion', {}).get('score', 0),
            'Quality Assessment': capabilities.get('task_completion', {}).get('assessment', 'unknown')
        })
        
        progress_bar.progress((i + 1) / min(dataset_size, len(sample_prompts)))
    
    status_text.text('Analysis complete!')
    
    # Display dataset analysis results
    dataset_df = pd.DataFrame(dataset_results)
    
    dataset_tab1, dataset_tab2, dataset_tab3 = st.tabs(["📈 Dataset Overview", "🔍 Distribution Analysis", "📋 Sample Details"])
    
    with dataset_tab1:
        st.subheader("Dataset Analysis Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(dataset_results))
        
        with col2:
            avg_entropy = dataset_df['Entropy'].mean()
            st.metric("Avg Entropy", f"{avg_entropy:.3f}")
        
        with col3:
            avg_length = dataset_df['Length'].mean()
            st.metric("Avg Length", f"{avg_length:.0f}")
        
        with col4:
            avg_task_completion = dataset_df['Task Completion'].mean()
            st.metric("Avg Task Completion", f"{avg_task_completion:.3f}")
        
        # Quality distribution
        quality_counts = dataset_df['Quality Assessment'].value_counts()
        fig_quality = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title="Quality Assessment Distribution"
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with dataset_tab2:
        st.subheader("Distribution Analysis")
        
        # Entropy distribution
        fig_entropy_dist = px.histogram(
            dataset_df,
            x='Entropy',
            nbins=20,
            title='Response Entropy Distribution',
            labels={'Entropy': 'Response Entropy', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_entropy_dist, use_container_width=True)
        
        # Length vs Task Completion scatter
        fig_scatter = px.scatter(
            dataset_df,
            x='Length',
            y='Task Completion',
            color='Quality Assessment',
            title='Response Length vs Task Completion',
            labels={'Length': 'Response Length (characters)', 'Task Completion': 'Task Completion Score'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        correlation_data = dataset_df[['Entropy', 'Length', 'Task Completion']].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix: Entropy, Length, Task Completion"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with dataset_tab3:
        st.subheader("Sample Details")
        
        # Display detailed results
        st.dataframe(
            dataset_df[['Sample', 'Prompt', 'Entropy', 'Length', 'Task Completion', 'Quality Assessment']],
            use_container_width=True
        )
        
        # Individual sample analysis
        selected_sample = st.selectbox("Select Sample for Detailed View", dataset_df['Sample'].tolist())
        
        if selected_sample:
            sample_data = dataset_df[dataset_df['Sample'] == selected_sample].iloc[0]
            
            st.subheader(f"Sample {selected_sample} Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Prompt:**")
                st.info(sample_data['Prompt'])
                
                st.markdown("**Metrics:**")
                st.write(f"• Entropy: {sample_data['Entropy']:.3f}")
                st.write(f"• Length: {sample_data['Length']:.0f} characters")
                st.write(f"• Task Completion: {sample_data['Task Completion']:.3f}")
                st.write(f"• Quality: {sample_data['Quality Assessment']}")
            
            with col2:
                st.markdown("**Response:**")
                st.write(sample_data['Response'])

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 <strong>Elmstash Demo</strong></p>
    <p>Observe • Evaluate • Discern</p>
</div>
""", unsafe_allow_html=True)

# Reset analysis state
if st.session_state.get('run_analysis', False):
    st.session_state.run_analysis = False
"""
Evaluation Pipeline - Orchestrates observation and evaluation workflow.

This is the main integration point that demonstrates the elmstash arch
between observation (what happened) and evaluation (how good was it).
"""

from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from observer.core import ModelObserver
from evaluator.core import ModelEvaluator


class EvaluationPipeline:
    """
    Orchestrates observation and evaluation workflow.
    
    This pipeline demonstrates the elmstash architecture:
    1. Observer: Records and measures what happened
    2. Evaluator: Makes judgments about how good it was
    3. Integration: Combines insights into actionable reports
    """
    
    def __init__(
        self, 
        db_path: str = "data/sessions.sqlite",
        evaluator_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            db_path: Path to database for storing observations
            evaluator_config: Configuration for evaluation judgments
        """
        self.observer = ModelObserver(db_path)
        self.evaluator = ModelEvaluator(evaluator_config)
        
        self.pipeline_history = []
    
    def run_full_analysis(self, session_id: str) -> Dict[str, Any]:
        """
        Run complete analysis pipeline for a session.
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Complete analysis results combining observation and evaluation
        """
        # Step 1: Observe and measure (objective)
        print("Step 1: Observing and measuring...")
        interactions = self.observer.get_session_data(session_id)
        
        if not interactions:
            return {'error': f'No data found for session {session_id}'}
        
        observed_metrics = self.observer.calculate_metrics(interactions)
        pattern_analysis = self.observer.analyze_patterns(interactions)
        
        # Step 2: Evaluate and judge (subjective)
        print("Step 2: Evaluating and judging...")
        observed_data = {
            'interactions': interactions,
            'metrics': observed_metrics,
            'patterns': pattern_analysis
        }
        
        evaluation_results = self.evaluator.evaluate_comprehensive(observed_data)
        
        # Step 3: Combine insights
        print("Step 3: Generating integrated report...")
        integrated_report = self.generate_integrated_report(
            session_id, observed_data, evaluation_results
        )
        
        # Store in pipeline history
        self.pipeline_history.append({
            'session_id': session_id,
            'timestamp': self._get_timestamp(),
            'results': integrated_report
        })
        
        return integrated_report
    
    def run_observation_only(self, session_id: str) -> Dict[str, Any]:
        """
        Run observation-only analysis (no judgments).
        
        Args:
            session_id: Session to observe
            
        Returns:
            Pure observation results without evaluation judgments
        """
        interactions = self.observer.get_session_data(session_id)
        
        if not interactions:
            return {'error': f'No data found for session {session_id}'}
        
        return {
            'session_id': session_id,
            'observation_type': 'objective_only',
            'interactions': interactions,
            'metrics': self.observer.calculate_metrics(interactions),
            'patterns': self.observer.analyze_patterns(interactions),
            'summary': {
                'total_interactions': len(interactions),
                'observation_timestamp': self._get_timestamp()
            }
        }
    
    def run_evaluation_only(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation-only analysis on pre-observed data.
        
        Args:
            observed_data: Previously collected observation data
            
        Returns:
            Pure evaluation results based on observed data
        """
        return {
            'evaluation_type': 'judgment_only',
            'evaluation_results': self.evaluator.evaluate_comprehensive(observed_data),
            'evaluation_timestamp': self._get_timestamp()
        }
    
    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple sessions using the pipeline.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Comparative analysis results
        """
        session_observations = {}
        
        # Observe all sessions
        for session_id in session_ids:
            interactions = self.observer.get_session_data(session_id)
            if interactions:
                session_observations[session_id] = {
                    'interactions': interactions,
                    'metrics': self.observer.calculate_metrics(interactions),
                    'patterns': self.observer.analyze_patterns(interactions)
                }
        
        if not session_observations:
            return {'error': 'No valid session data found'}
        
        # Comparative evaluation
        comparative_results = self.evaluator.compare_models(session_observations)
        
        return {
            'comparison_type': 'multi_session',
            'sessions_compared': list(session_observations.keys()),
            'individual_observations': session_observations,
            'comparative_evaluation': comparative_results,
            'comparison_timestamp': self._get_timestamp()
        }
    
    def record_and_analyze_interaction(
        self,
        session_id: str,
        step: int,
        input_str: str,
        output_str: str,
        action: str = "query",
        metadata: Optional[Dict[str, Any]] = None,
        immediate_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Record a new interaction and optionally analyze immediately.
        
        Args:
            session_id: Session identifier
            step: Step number
            input_str: Input prompt
            output_str: Model response
            action: Action type
            metadata: Additional metadata
            immediate_analysis: Whether to run analysis immediately
            
        Returns:
            Recording confirmation and optional analysis results
        """
        # Step 1: Record the interaction (observation)
        self.observer.record_interaction(
            session_id, step, input_str, output_str, action, metadata
        )
        
        result = {
            'recording_status': 'success',
            'session_id': session_id,
            'step': step,
            'timestamp': self._get_timestamp()
        }
        
        # Step 2: Optional immediate analysis
        if immediate_analysis:
            # Get recent interactions for context
            interactions = self.observer.get_session_data(session_id)
            recent_interactions = interactions[-5:]  # Last 5 interactions
            
            if len(recent_interactions) >= 1:
                observed_data = {
                    'interactions': recent_interactions,
                    'metrics': self.observer.calculate_metrics(recent_interactions),
                    'patterns': self.observer.analyze_patterns(recent_interactions)
                }
                
                evaluation = self.evaluator.evaluate_comprehensive(observed_data)
                
                result['immediate_analysis'] = {
                    'observed_data': observed_data,
                    'evaluation': evaluation
                }
        
        return result
    
    def generate_integrated_report(
        self,
        session_id: str,
        observed_data: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate integrated report combining observation and evaluation.
        
        Args:
            session_id: Session identifier
            observed_data: Objective observation data
            evaluation_results: Subjective evaluation results
            
        Returns:
            Integrated analysis report
        """
        interactions = observed_data.get('interactions', [])
        metrics = observed_data.get('metrics', {})
        patterns = observed_data.get('patterns', {})
        
        capabilities = evaluation_results.get('capabilities', {})
        alignment = evaluation_results.get('alignment', {})
        overall_assessment = evaluation_results.get('overall_assessment', {})
        
        return {
            'session_id': session_id,
            'analysis_type': 'integrated_observation_evaluation',
            'timestamp': self._get_timestamp(),
            
            # Objective observations (what happened)
            'observations': {
                'interaction_count': len(interactions),
                'objective_metrics': metrics,
                'detected_patterns': patterns,
                'data_quality': self._assess_data_quality(observed_data)
            },
            
            # Subjective evaluations (how good was it)
            'evaluations': {
                'capability_assessment': capabilities,
                'alignment_assessment': alignment,
                'overall_performance': overall_assessment
            },
            
            # Integrated insights
            'insights': {
                'key_findings': self._extract_key_findings(observed_data, evaluation_results),
                'warning_signs': self._identify_warning_signs(observed_data, evaluation_results),
                'recommendations': self._generate_recommendations(observed_data, evaluation_results),
                'confidence_level': self._assess_confidence_level(observed_data)
            },
            
            # Metadata
            'metadata': {
                'observer_version': 'v1.0',
                'evaluator_version': 'v1.0',
                'pipeline_version': 'v1.0',
                'analysis_completeness': self._assess_analysis_completeness(observed_data, evaluation_results)
            }
        }
    
    def get_pipeline_history(self) -> List[Dict[str, Any]]:
        """Get history of pipeline runs."""
        return self.pipeline_history
    
    def clear_pipeline_history(self):
        """Clear pipeline history."""
        self.pipeline_history = []
    
    # Private helper methods
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _assess_data_quality(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of observed data."""
        interactions = observed_data.get('interactions', [])
        metrics = observed_data.get('metrics', {})
        
        quality_score = 0.0
        quality_factors = []
        
        # Sufficient sample size
        if len(interactions) >= 5:
            quality_score += 0.3
            quality_factors.append("Sufficient sample size")
        elif len(interactions) >= 2:
            quality_score += 0.15
            quality_factors.append("Minimal sample size")
        
        # Metrics availability
        if metrics:
            quality_score += 0.3
            quality_factors.append("Metrics calculated")
        
        # Response quality
        responses = [i.get('output', '') for i in interactions]
        if responses and all(len(r) > 10 for r in responses):
            quality_score += 0.2
            quality_factors.append("Substantial responses")
        
        # Input diversity
        inputs = [i.get('input', '') for i in interactions]
        if len(set(inputs)) > len(inputs) * 0.7:  # 70% unique inputs
            quality_score += 0.2
            quality_factors.append("Diverse inputs")
        
        return {
            'quality_score': quality_score,
            'quality_level': 'high' if quality_score >= 0.8 else 'medium' if quality_score >= 0.5 else 'low',
            'quality_factors': quality_factors
        }
    
    def _extract_key_findings(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from combined analysis."""
        findings = []
        
        # From observations
        metrics = observed_data.get('metrics', {})
        if metrics.get('response_entropy', 0) < 1.0:
            findings.append("Low response diversity detected (possible mode collapse)")
        
        # From evaluations
        overall = evaluation_results.get('overall_assessment', {})
        if overall.get('quality_rating') == 'excellent':
            findings.append("Model demonstrates excellent overall performance")
        elif overall.get('quality_rating') == 'poor':
            findings.append("Model performance is below acceptable standards")
        
        capabilities = evaluation_results.get('capabilities', {})
        if isinstance(capabilities, dict):
            for capability, result in capabilities.items():
                if isinstance(result, dict) and result.get('assessment') == 'excellent':
                    findings.append(f"Strong performance in {capability.replace('_', ' ')}")
        
        return findings
    
    def _identify_warning_signs(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> List[str]:
        """Identify warning signs from analysis."""
        warnings = []
        
        # Observation-based warnings
        metrics = observed_data.get('metrics', {})
        if metrics.get('response_entropy', 1.0) < 0.5:
            warnings.append("Very low response entropy - possible mode collapse")
        
        patterns = observed_data.get('patterns', {})
        repetition_analysis = patterns.get('repetition_analysis', {})
        if repetition_analysis.get('repetition_ratio', 0) > 0.5:
            warnings.append("High repetition rate in responses")
        
        # Evaluation-based warnings
        overall = evaluation_results.get('overall_assessment', {})
        if overall.get('overall_score', 0) < 0.4:
            warnings.append("Overall performance score below acceptable threshold")
        
        alignment = evaluation_results.get('alignment', {})
        if isinstance(alignment, dict):
            for metric, result in alignment.items():
                if isinstance(result, dict) and result.get('score', 1.0) < 0.6:
                    warnings.append(f"Low {metric.replace('_', ' ')} score")
        
        return warnings
    
    def _generate_recommendations(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on observations
        metrics = observed_data.get('metrics', {})
        if metrics.get('response_entropy', 1.0) < 1.0:
            recommendations.append("Increase temperature or adjust sampling parameters to improve response diversity")
        
        # Based on evaluations
        overall = evaluation_results.get('overall_assessment', {})
        if 'recommendations' in overall:
            recommendations.extend(overall['recommendations'])
        
        capabilities = evaluation_results.get('capabilities', {})
        if isinstance(capabilities, dict):
            for capability, result in capabilities.items():
                if isinstance(result, dict) and 'improvement_suggestions' in result:
                    recommendations.extend(result['improvement_suggestions'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_confidence_level(self, observed_data: Dict[str, Any]) -> str:
        """Assess confidence level in the analysis."""
        interactions = observed_data.get('interactions', [])
        
        if len(interactions) >= 10:
            return 'high'
        elif len(interactions) >= 5:
            return 'medium'
        elif len(interactions) >= 2:
            return 'low'
        else:
            return 'very_low'
    
    def _assess_analysis_completeness(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess completeness of the analysis."""
        completeness = {
            'has_observations': bool(observed_data.get('interactions')),
            'has_metrics': bool(observed_data.get('metrics')),
            'has_patterns': bool(observed_data.get('patterns')),
            'has_capability_eval': bool(evaluation_results.get('capabilities')),
            'has_alignment_eval': bool(evaluation_results.get('alignment')),
            'has_overall_assessment': bool(evaluation_results.get('overall_assessment'))
        }
        
        completeness_score = sum(completeness.values()) / len(completeness)
        
        return {
            'individual_components': completeness,
            'completeness_score': completeness_score,
            'completeness_level': 'complete' if completeness_score >= 0.8 else 'partial'
        }
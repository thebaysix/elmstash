"""
Observation Pipeline - Pure observation workflow without evaluation.

This pipeline demonstrates observation-only analysis, focusing purely
on recording and measuring what happened without making judgments.
"""

from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from observer.core import ModelObserver


class ObservationPipeline:
    """
    Pure observation pipeline without evaluation judgments.
    
    This pipeline demonstrates the observer-only workflow:
    1. Record interactions
    2. Calculate objective metrics
    3. Detect statistical patterns
    4. Generate observation reports (no quality judgments)
    """
    
    def __init__(self, db_path: str = "data/sessions.sqlite"):
        """
        Initialize the observation pipeline.
        
        Args:
            db_path: Path to database for storing observations
        """
        self.observer = ModelObserver(db_path)
        self.observation_history = []
    
    def run_observation_analysis(self, session_id: str) -> Dict[str, Any]:
        """
        Run pure observation analysis for a session.
        
        Args:
            session_id: Session to observe
            
        Returns:
            Pure observation results without any quality judgments
        """
        # Step 1: Retrieve recorded interactions
        print("Step 1: Retrieving recorded interactions...")
        interactions = self.observer.get_session_data(session_id)
        
        if not interactions:
            return {'error': f'No data found for session {session_id}'}
        
        # Step 2: Calculate objective metrics
        print("Step 2: Calculating objective metrics...")
        metrics = self.observer.calculate_metrics(interactions)
        
        # Step 3: Detect statistical patterns
        print("Step 3: Detecting statistical patterns...")
        patterns = self.observer.analyze_patterns(interactions)
        
        # Step 4: Generate observation report
        print("Step 4: Generating observation report...")
        observation_report = self.generate_observation_report(
            session_id, interactions, metrics, patterns
        )
        
        # Store in observation history
        self.observation_history.append({
            'session_id': session_id,
            'timestamp': self._get_timestamp(),
            'results': observation_report
        })
        
        return observation_report
    
    def record_and_observe(
        self,
        session_id: str,
        step: int,
        input_str: str,
        output_str: str,
        action: str = "query",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record an interaction and provide immediate observation data.
        
        Args:
            session_id: Session identifier
            step: Step number
            input_str: Input prompt
            output_str: Model response
            action: Action type
            metadata: Additional metadata
            
        Returns:
            Recording confirmation and immediate observation data
        """
        # Record the interaction
        self.observer.record_interaction(
            session_id, step, input_str, output_str, action, metadata
        )
        
        # Get immediate observation data
        recent_interactions = self.observer.get_session_data(session_id)
        
        # Calculate metrics for recent context (last 3 interactions)
        context_interactions = recent_interactions[-3:] if len(recent_interactions) >= 3 else recent_interactions
        
        immediate_metrics = self.observer.calculate_metrics(context_interactions)
        immediate_patterns = self.observer.analyze_patterns(context_interactions)
        
        return {
            'recording_status': 'success',
            'session_id': session_id,
            'step': step,
            'timestamp': self._get_timestamp(),
            'immediate_observations': {
                'context_size': len(context_interactions),
                'metrics': immediate_metrics,
                'patterns': immediate_patterns
            }
        }
    
    def compare_sessions_objectively(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple sessions using only objective measurements.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Objective comparison without quality judgments
        """
        session_observations = {}
        
        # Observe all sessions
        for session_id in session_ids:
            interactions = self.observer.get_session_data(session_id)
            if interactions:
                session_observations[session_id] = {
                    'interactions': interactions,
                    'metrics': self.observer.calculate_metrics(interactions),
                    'patterns': self.observer.analyze_patterns(interactions),
                    'summary': self.observer.get_session_summary(session_id)
                }
        
        if not session_observations:
            return {'error': 'No valid session data found'}
        
        # Generate objective comparison
        objective_comparison = self._generate_objective_comparison(session_observations)
        
        return {
            'comparison_type': 'objective_only',
            'sessions_compared': list(session_observations.keys()),
            'individual_observations': session_observations,
            'objective_comparison': objective_comparison,
            'comparison_timestamp': self._get_timestamp()
        }
    
    def generate_observation_report(
        self,
        session_id: str,
        interactions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate pure observation report without judgments.
        
        Args:
            session_id: Session identifier
            interactions: List of interactions
            metrics: Calculated metrics
            patterns: Detected patterns
            
        Returns:
            Pure observation report
        """
        return {
            'session_id': session_id,
            'report_type': 'pure_observation',
            'timestamp': self._get_timestamp(),
            
            # Raw data summary
            'data_summary': {
                'total_interactions': len(interactions),
                'time_span': self._calculate_time_span(interactions),
                'data_completeness': self._assess_data_completeness(interactions)
            },
            
            # Objective measurements
            'measurements': {
                'entropy_metrics': self._extract_entropy_metrics(metrics),
                'length_statistics': self._extract_length_statistics(metrics),
                'diversity_measures': self._extract_diversity_measures(metrics),
                'timing_data': metrics.get('timing_stats', {})
            },
            
            # Statistical patterns
            'detected_patterns': {
                'repetition_analysis': patterns.get('repetition_patterns', {}),
                'structural_analysis': patterns.get('structural_patterns', {}),
                'consistency_analysis': patterns.get('consistency_patterns', {}),
                'content_analysis': patterns.get('content_patterns', {})
            },
            
            # Metadata
            'observation_metadata': {
                'observer_version': 'v1.0',
                'analysis_completeness': self._calculate_analysis_completeness(metrics, patterns),
                'data_quality_indicators': self._extract_data_quality_indicators(interactions, metrics)
            }
        }
    
    def get_observation_history(self) -> List[Dict[str, Any]]:
        """Get history of observation runs."""
        return self.observation_history
    
    def clear_observation_history(self):
        """Clear observation history."""
        self.observation_history = []
    
    # Private helper methods
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_time_span(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate time span of interactions."""
        if not interactions:
            return {'no_data': True}
        
        timestamps = [i.get('timestamp') for i in interactions if i.get('timestamp')]
        
        if not timestamps:
            return {'no_timestamp_data': True}
        
        return {
            'no_data': False,
            'no_timestamp_data': False,
            'first_interaction': min(timestamps),
            'last_interaction': max(timestamps),
            'total_interactions': len(interactions)
        }
    
    def _assess_data_completeness(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess completeness of interaction data."""
        if not interactions:
            return {'no_data': True}
        
        completeness_indicators = {
            'has_inputs': sum(1 for i in interactions if i.get('input')),
            'has_outputs': sum(1 for i in interactions if i.get('output')),
            'has_metadata': sum(1 for i in interactions if i.get('metadata')),
            'has_timestamps': sum(1 for i in interactions if i.get('timestamp')),
            'total_interactions': len(interactions)
        }
        
        # Calculate completeness ratios
        total = completeness_indicators['total_interactions']
        completeness_ratios = {
            'input_completeness': completeness_indicators['has_inputs'] / total,
            'output_completeness': completeness_indicators['has_outputs'] / total,
            'metadata_completeness': completeness_indicators['has_metadata'] / total,
            'timestamp_completeness': completeness_indicators['has_timestamps'] / total
        }
        
        return {
            'no_data': False,
            'completeness_indicators': completeness_indicators,
            'completeness_ratios': completeness_ratios,
            'overall_completeness': sum(completeness_ratios.values()) / len(completeness_ratios)
        }
    
    def _extract_entropy_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entropy-related metrics."""
        return {
            'response_entropy': metrics.get('response_entropy', 0),
            'input_entropy': metrics.get('input_entropy', 0),
            'character_entropy_stats': metrics.get('character_entropy_stats', {}),
            'uniqueness_ratios': {
                'response_uniqueness': metrics.get('response_uniqueness_ratio', 0),
                'input_uniqueness': metrics.get('input_uniqueness_ratio', 0)
            }
        }
    
    def _extract_length_statistics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract length-related statistics."""
        return {
            'response_lengths': metrics.get('response_length_stats', {}),
            'input_lengths': metrics.get('input_length_stats', {}),
            'token_counts': metrics.get('token_count_stats', {})
        }
    
    def _extract_diversity_measures(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract diversity-related measures."""
        return {
            'unique_responses': metrics.get('unique_responses', 0),
            'unique_inputs': metrics.get('unique_inputs', 0),
            'total_interactions': metrics.get('interaction_count', 0),
            'diversity_ratios': {
                'response_diversity': metrics.get('response_uniqueness_ratio', 0),
                'input_diversity': metrics.get('input_uniqueness_ratio', 0)
            }
        }
    
    def _calculate_analysis_completeness(self, metrics: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate completeness of the analysis."""
        analysis_components = {
            'has_entropy_metrics': bool(metrics.get('response_entropy')),
            'has_length_stats': bool(metrics.get('response_length_stats')),
            'has_diversity_measures': bool(metrics.get('unique_responses')),
            'has_pattern_analysis': bool(patterns.get('repetition_patterns')),
            'has_consistency_analysis': bool(patterns.get('consistency_patterns')),
            'has_timing_data': bool(metrics.get('timing_stats'))
        }
        
        completeness_score = sum(analysis_components.values()) / len(analysis_components)
        
        return {
            'individual_components': analysis_components,
            'completeness_score': completeness_score,
            'analysis_level': 'complete' if completeness_score >= 0.8 else 'partial'
        }
    
    def _extract_data_quality_indicators(
        self, 
        interactions: List[Dict[str, Any]], 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract indicators of data quality."""
        if not interactions:
            return {'no_data': True}
        
        # Check for empty responses
        empty_responses = sum(1 for i in interactions if not i.get('output', '').strip())
        
        # Check for very short responses
        short_responses = sum(1 for i in interactions if len(i.get('output', '')) < 10)
        
        # Check for duplicate responses
        responses = [i.get('output', '') for i in interactions]
        duplicate_responses = len(responses) - len(set(responses))
        
        return {
            'no_data': False,
            'total_interactions': len(interactions),
            'empty_responses': empty_responses,
            'short_responses': short_responses,
            'duplicate_responses': duplicate_responses,
            'quality_ratios': {
                'empty_ratio': empty_responses / len(interactions),
                'short_ratio': short_responses / len(interactions),
                'duplicate_ratio': duplicate_responses / len(interactions)
            }
        }
    
    def _generate_objective_comparison(self, session_observations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate objective comparison between sessions."""
        comparison_metrics = {}
        
        for session_id, observation_data in session_observations.items():
            metrics = observation_data.get('metrics', {})
            summary = observation_data.get('summary', {})
            
            comparison_metrics[session_id] = {
                'interaction_count': summary.get('interaction_count', 0),
                'avg_response_length': summary.get('avg_response_length', 0),
                'unique_responses': summary.get('unique_responses', 0),
                'response_entropy': metrics.get('response_entropy', 0),
                'response_uniqueness_ratio': metrics.get('response_uniqueness_ratio', 0)
            }
        
        # Calculate comparative statistics
        all_sessions = list(session_observations.keys())
        comparative_stats = {}
        
        if len(all_sessions) >= 2:
            # Find sessions with highest/lowest values for each metric
            for metric_name in ['interaction_count', 'avg_response_length', 'response_entropy']:
                values = [(session, comparison_metrics[session].get(metric_name, 0)) 
                         for session in all_sessions]
                
                highest = max(values, key=lambda x: x[1])
                lowest = min(values, key=lambda x: x[1])
                
                comparative_stats[metric_name] = {
                    'highest': {'session': highest[0], 'value': highest[1]},
                    'lowest': {'session': lowest[0], 'value': lowest[1]},
                    'range': highest[1] - lowest[1]
                }
        
        return {
            'individual_metrics': comparison_metrics,
            'comparative_statistics': comparative_stats,
            'sessions_count': len(all_sessions)
        }
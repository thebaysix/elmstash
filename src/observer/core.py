"""
Observer Core - Passively observes and measures model interactions.

This module provides the core ModelObserver class that records interactions
and calculates objective metrics without making judgments about quality.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import os

from .logging.logger import init_db, log_interaction
from .metrics.entropy import calc_entropy, calc_character_entropy, calc_response_entropy
from .analysis.patterns import PatternAnalyzer


class ModelObserver:
    """
    Passively observes and measures model interactions.
    
    The Observer follows the principle: "What happened?" - it records
    objective data and calculates mathematical metrics without making
    judgments about quality or performance.
    """
    
    def __init__(self, db_path: str = "data/sessions.sqlite"):
        """
        Initialize the model observer.
        
        Args:
            db_path: Path to SQLite database for storing observations
        """
        self.db_path = db_path
        self._local = threading.local()
        
        # Initialize database with custom path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        # Initialize the database schema (using a temporary connection)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    session_id TEXT,
                    step INTEGER,
                    timestamp TEXT,
                    input TEXT,
                    action TEXT,
                    output TEXT,
                    metadata TEXT
                )
            ''')
            conn.commit()
        
        self.pattern_analyzer = PatternAnalyzer()
    
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn
    
    def record_interaction(
        self,
        session_id: str,
        step: int,
        input_str: str,
        output_str: str,
        action: str = "query",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a model interaction.
        
        Args:
            session_id: Session identifier
            step: Step number in the session
            input_str: Input prompt
            output_str: Model response
            action: Action type (default: "query")
            metadata: Additional metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        import json
        meta_json = json.dumps(metadata or {})
        
        cursor.execute('''
            INSERT INTO interactions (session_id, step, timestamp, input, action, output, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, step, timestamp, input_str, action, output_str, meta_json))
        conn.commit()
    
    def get_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all interactions for a session.
        
        Args:
            session_id: Session to retrieve
            
        Returns:
            List of interaction dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT session_id, step, input, action, output, metadata, timestamp
        FROM interactions 
        WHERE session_id = ?
        ORDER BY step
        ''', (session_id,))
        
        rows = cursor.fetchall()
        
        interactions = []
        for row in rows:
            interaction = {
                'session_id': row[0],
                'step': row[1],
                'input': row[2],
                'action': row[3],
                'output': row[4],
                'metadata': self._parse_metadata(row[5]),
                'timestamp': row[6]
            }
            interactions.append(interaction)
        
        return interactions
    
    def calculate_metrics(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate objective metrics from interactions.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dictionary of calculated metrics
        """
        if not interactions:
            return {'error': 'No interactions provided'}
        
        # Extract responses and inputs
        responses = [i.get('output', '') for i in interactions]
        inputs = [i.get('input', '') for i in interactions]
        
        # Calculate entropy metrics
        metrics = {
            'response_entropy': calc_response_entropy(responses),
            'input_entropy': calc_response_entropy(inputs),  # Reuse same function
            'character_entropy_stats': self._calculate_character_entropy_stats(responses),
            'token_count_stats': self._calculate_token_count_stats(responses),
            'response_length_stats': self._calculate_length_stats(responses),
            'input_length_stats': self._calculate_length_stats(inputs),
            'interaction_count': len(interactions),
            'unique_responses': len(set(responses)),
            'unique_inputs': len(set(inputs)),
            'response_uniqueness_ratio': len(set(responses)) / len(responses) if responses else 0,
            'input_uniqueness_ratio': len(set(inputs)) / len(inputs) if inputs else 0
        }
        
        # Add timing metrics if available
        timing_metrics = self._calculate_timing_metrics(interactions)
        if timing_metrics:
            metrics['timing_stats'] = timing_metrics
        
        return metrics
    
    def analyze_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify statistical patterns in the interaction data.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dictionary of detected patterns
        """
        if not interactions:
            return {'error': 'No interactions provided'}
        
        responses = [i.get('output', '') for i in interactions]
        inputs = [i.get('input', '') for i in interactions]
        
        patterns = {
            'repetition_patterns': self.pattern_analyzer.detect_repetition_patterns(responses),
            'length_patterns': self.pattern_analyzer.detect_length_patterns(responses),
            'structural_patterns': self.pattern_analyzer.detect_structural_patterns(responses),
            'content_patterns': self.pattern_analyzer.detect_content_patterns(responses),
            'input_patterns': {
                'length_patterns': self.pattern_analyzer.detect_length_patterns(inputs),
                'content_patterns': self.pattern_analyzer.detect_content_patterns(inputs)
            },
            'consistency_patterns': self._analyze_consistency_patterns(interactions)
        }
        
        return patterns
    
    def get_all_sessions(self) -> List[str]:
        """
        Get list of all session IDs in the database.
        
        Returns:
            List of session IDs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT session_id FROM interactions ORDER BY session_id')
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of a session without detailed analysis.
        
        Args:
            session_id: Session to summarize
            
        Returns:
            Dictionary with session summary
        """
        interactions = self.get_session_data(session_id)
        
        if not interactions:
            return {'error': f'No data found for session {session_id}'}
        
        responses = [i.get('output', '') for i in interactions]
        inputs = [i.get('input', '') for i in interactions]
        
        return {
            'session_id': session_id,
            'interaction_count': len(interactions),
            'first_interaction': interactions[0]['timestamp'],
            'last_interaction': interactions[-1]['timestamp'],
            'total_input_chars': sum(len(inp) for inp in inputs),
            'total_output_chars': sum(len(resp) for resp in responses),
            'avg_response_length': sum(len(resp) for resp in responses) / len(responses),
            'unique_responses': len(set(responses)),
            'unique_inputs': len(set(inputs))
        }
    
    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')
    
    # Private helper methods
    
    def _parse_metadata(self, metadata_str: Optional[str]) -> Dict[str, Any]:
        """Parse metadata string back to dictionary."""
        if not metadata_str:
            return {}
        
        try:
            import json
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _calculate_character_entropy_stats(self, responses: List[str]) -> Dict[str, float]:
        """Calculate character entropy statistics for responses."""
        if not responses:
            return {'mean': 0, 'min': 0, 'max': 0}
        
        entropies = [calc_character_entropy(response) for response in responses]
        
        return {
            'mean': sum(entropies) / len(entropies),
            'min': min(entropies),
            'max': max(entropies),
            'std': self._calculate_std(entropies)
        }
    
    def _calculate_token_count_stats(self, responses: List[str]) -> Dict[str, float]:
        """Calculate token count statistics (simplified word count)."""
        if not responses:
            return {'mean': 0, 'min': 0, 'max': 0}
        
        # Simple word count as token approximation
        token_counts = [len(response.split()) for response in responses]
        
        return {
            'mean': sum(token_counts) / len(token_counts),
            'min': min(token_counts),
            'max': max(token_counts),
            'std': self._calculate_std(token_counts)
        }
    
    def _calculate_length_stats(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length statistics for texts."""
        if not texts:
            return {'mean': 0, 'min': 0, 'max': 0}
        
        lengths = [len(text) for text in texts]
        
        return {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'std': self._calculate_std(lengths)
        }
    
    def _calculate_timing_metrics(self, interactions: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Calculate timing metrics if timing data is available."""
        response_times = []
        
        for interaction in interactions:
            metadata = interaction.get('metadata', {})
            if 'response_time' in metadata:
                response_times.append(metadata['response_time'])
        
        if not response_times:
            return None
        
        return {
            'mean_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'std_response_time': self._calculate_std(response_times)
        }
    
    def _analyze_consistency_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency patterns across interactions."""
        if len(interactions) < 2:
            return {'insufficient_data': True}
        
        responses = [i.get('output', '') for i in interactions]
        
        # Response length consistency
        lengths = [len(response) for response in responses]
        length_variance = self._calculate_variance(lengths)
        
        # Response uniqueness
        unique_responses = len(set(responses))
        uniqueness_ratio = unique_responses / len(responses)
        
        return {
            'insufficient_data': False,
            'length_variance': length_variance,
            'uniqueness_ratio': uniqueness_ratio,
            'repetition_ratio': 1 - uniqueness_ratio,
            'consistency_score': 1 - (length_variance / (sum(lengths) / len(lengths))) if lengths else 0
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
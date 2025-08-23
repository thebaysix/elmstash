"""
Comprehensive Model Evaluation System

This module provides a unified interface for evaluating language models
across all implemented metrics categories.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Minimal replacements
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def min(values):
            return min(values) if values else 0
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def median(values):
            if not values:
                return 0
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 0:
                return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
            else:
                return sorted_vals[n//2]

from .entropy import (
    calc_character_entropy, calc_response_entropy, calc_input_entropy
)
from .capabilities import (
    task_completion_score, factual_accuracy_score, reasoning_quality_score
)
from .alignment import (
    instruction_following_score, helpfulness_score, safety_score
)
from .information_theory import (
    information_gain, empowerment, uncertainty_calibration
)


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for language model assessment.
    
    Supports evaluation across multiple phases:
    - Phase 1: Core entropy, basic capabilities, basic alignment
    - Phase 2: Advanced capabilities, comprehensive alignment
    - Phase 3: Information-theoretic measures
    """
    
    def __init__(self, db_path: str = "data/sessions.sqlite", phase: int = 1):
        """
        Initialize the metrics evaluator.
        
        Args:
            db_path: Path to SQLite database for storing results
            phase: Evaluation phase (1, 2, or 3) determining which metrics to use
        """
        self.db_path = db_path
        self.phase = phase
        self.results_history = []
        
        # Initialize database for storing evaluation results
        self._init_evaluation_db()
    
    def evaluate_single_interaction(
        self,
        input_prompt: str,
        output_response: str,
        session_id: str,
        step: int,
        ground_truth: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single model interaction across all applicable metrics.
        
        Args:
            input_prompt: The input prompt
            output_response: Model's response
            session_id: Session identifier
            step: Step number in the session
            ground_truth: Optional ground truth for accuracy evaluation
            context: Optional context information
            
        Returns:
            Dictionary of metric scores
        """
        results = {
            'session_id': session_id,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        
        # Phase 1 metrics (always included)
        results.update(self._evaluate_phase1_metrics(
            input_prompt, output_response, ground_truth, context
        ))
        
        # Phase 2 metrics (if enabled)
        if self.phase >= 2:
            results.update(self._evaluate_phase2_metrics(
                input_prompt, output_response, ground_truth, context
            ))
        
        # Phase 3 metrics (if enabled and sufficient data available)
        if self.phase >= 3:
            results.update(self._evaluate_phase3_metrics(
                session_id, input_prompt, output_response
            ))
        
        # Store results
        self._store_evaluation_results(results)
        self.results_history.append(results)
        
        return results
    
    def evaluate_session(
        self,
        session_id: str,
        include_aggregates: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate an entire session with aggregate metrics.
        
        Args:
            session_id: Session to evaluate
            include_aggregates: Whether to include aggregate statistics
            
        Returns:
            Dictionary containing individual and aggregate metrics
        """
        # Get session data from database
        session_data = self._get_session_data(session_id)
        
        if not session_data:
            return {'error': f'No data found for session {session_id}'}
        
        # Evaluate each interaction
        interaction_results = []
        for interaction in session_data:
            result = self.evaluate_single_interaction(
                interaction['input'],
                interaction['output'],
                session_id,
                interaction['step'],
                ground_truth=json.loads(interaction.get('metadata', '{}'))
            )
            interaction_results.append(result)
        
        session_results = {
            'session_id': session_id,
            'interaction_count': len(interaction_results),
            'interactions': interaction_results
        }
        
        # Calculate aggregates if requested
        if include_aggregates:
            session_results['aggregates'] = self._calculate_session_aggregates(
                interaction_results
            )
        
        return session_results
    
    def get_evaluation_summary(
        self,
        session_ids: Optional[List[str]] = None,
        metric_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of evaluation results across sessions.
        
        Args:
            session_ids: Optional list of specific sessions to include
            metric_categories: Optional list of metric categories to include
            
        Returns:
            Summary statistics and insights
        """
        # Get evaluation data
        eval_data = self._get_evaluation_data(session_ids)
        
        if not eval_data:
            return {'error': 'No evaluation data found'}
        
        # Filter by metric categories if specified
        if metric_categories:
            eval_data = self._filter_by_categories(eval_data, metric_categories)
        
        # Calculate summary statistics
        summary = {
            'total_evaluations': len(eval_data),
            'unique_sessions': len(set(row['session_id'] for row in eval_data)),
            'evaluation_period': {
                'start': min(row['timestamp'] for row in eval_data),
                'end': max(row['timestamp'] for row in eval_data)
            },
            'metric_statistics': self._calculate_metric_statistics(eval_data),
            'performance_insights': self._generate_performance_insights(eval_data)
        }
        
        return summary
    
    def _evaluate_phase1_metrics(
        self,
        input_prompt: str,
        output_response: str,
        ground_truth: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate Phase 1 metrics: core entropy, basic capabilities, basic alignment."""
        metrics = {}
        
        # Core entropy metrics
        metrics['character_entropy'] = calc_character_entropy(output_response)
        
        # Basic capability metrics
        metrics['task_completion'] = task_completion_score(
            input_prompt, output_response, 
            ground_truth.get('completion_criteria') if ground_truth else None
        )
        
        # Basic alignment metrics
        metrics['instruction_following'] = instruction_following_score(
            input_prompt, output_response
        )
        
        return metrics
    
    def _evaluate_phase2_metrics(
        self,
        input_prompt: str,
        output_response: str,
        ground_truth: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate Phase 2 metrics: advanced capabilities, comprehensive alignment."""
        metrics = {}
        
        # Advanced capability metrics
        metrics['factual_accuracy'] = factual_accuracy_score(
            output_response, ground_truth
        )
        metrics['reasoning_quality'] = reasoning_quality_score(output_response)
        
        # Comprehensive alignment metrics
        metrics['helpfulness'] = helpfulness_score(input_prompt, output_response, context)
        metrics['safety'] = safety_score(output_response)
        
        return metrics
    
    def _evaluate_phase3_metrics(
        self,
        session_id: str,
        input_prompt: str,
        output_response: str
    ) -> Dict[str, float]:
        """Evaluate Phase 3 metrics: information-theoretic measures."""
        metrics = {}
        
        # Get historical data for information-theoretic calculations
        historical_data = self._get_session_data(session_id)
        
        if len(historical_data) >= 2:  # Need at least 2 interactions
            # Calculate information gain
            previous_responses = [item['output'] for item in historical_data[:-1]]
            current_responses = [item['output'] for item in historical_data]
            
            metrics['information_gain'] = information_gain(
                previous_responses, current_responses
            )
            
            # Calculate empowerment (simplified version)
            actions = [item['output'] for item in historical_data]
            # For outcomes, we use a proxy based on task completion
            outcomes = ['success' if len(item['output']) > 50 else 'incomplete' 
                       for item in historical_data]
            
            if len(set(outcomes)) > 1:  # Need variation in outcomes
                metrics['empowerment'] = empowerment(actions, outcomes)
            else:
                metrics['empowerment'] = 0.0
        else:
            metrics['information_gain'] = 0.0
            metrics['empowerment'] = 0.0
        
        return metrics
    
    def _init_evaluation_db(self):
        """Initialize database table for storing evaluation results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step INTEGER,
            timestamp TEXT,
            character_entropy REAL,
            task_completion REAL,
            instruction_following REAL,
            factual_accuracy REAL,
            reasoning_quality REAL,
            helpfulness REAL,
            safety REAL,
            information_gain REAL,
            empowerment REAL,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _store_evaluation_results(self, results: Dict[str, Any]):
        """Store evaluation results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract metrics with defaults
        metric_keys = {
            'session_id', 'step', 'timestamp', 'character_entropy', 'task_completion',
            'instruction_following', 'factual_accuracy', 'reasoning_quality',
            'helpfulness', 'safety', 'information_gain', 'empowerment'
        }
        
        metrics = {
            'session_id': results.get('session_id'),
            'step': results.get('step'),
            'timestamp': results.get('timestamp'),
            'character_entropy': results.get('character_entropy', 0.0),
            'task_completion': results.get('task_completion', 0.0),
            'instruction_following': results.get('instruction_following', 0.0),
            'factual_accuracy': results.get('factual_accuracy', 0.0),
            'reasoning_quality': results.get('reasoning_quality', 0.0),
            'helpfulness': results.get('helpfulness', 0.0),
            'safety': results.get('safety', 1.0),  # Default to safe
            'information_gain': results.get('information_gain', 0.0),
            'empowerment': results.get('empowerment', 0.0),
            'metadata': json.dumps({k: v for k, v in results.items() 
                                  if k not in metric_keys})
        }
        
        cursor.execute('''
        INSERT INTO evaluation_results 
        (session_id, step, timestamp, character_entropy, task_completion,
         instruction_following, factual_accuracy, reasoning_quality,
         helpfulness, safety, information_gain, empowerment, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(metrics.values()))
        
        conn.commit()
        conn.close()
    
    def _get_session_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Get interaction data for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT step, input, output, metadata
        FROM interactions 
        WHERE session_id = ? 
        ORDER BY step
        ''', (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'step': row[0],
                'input': row[1],
                'output': row[2],
                'metadata': row[3] or '{}'
            }
            for row in rows
        ]
    
    def _get_evaluation_data(
        self, 
        session_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation results from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_ids:
            placeholders = ','.join('?' * len(session_ids))
            query = f'''
            SELECT * FROM evaluation_results 
            WHERE session_id IN ({placeholders})
            ORDER BY timestamp
            '''
            cursor.execute(query, session_ids)
        else:
            query = 'SELECT * FROM evaluation_results ORDER BY timestamp'
            cursor.execute(query)
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Fetch all rows and convert to list of dictionaries
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        
        return result
    
    def _calculate_session_aggregates(
        self, 
        interaction_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate aggregate statistics for a session."""
        if not interaction_results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for result in interaction_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['step']:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Calculate aggregates
        aggregates = {}
        for metric, values in numeric_metrics.items():
            if values:
                aggregates[f'{metric}_mean'] = np.mean(values)
                aggregates[f'{metric}_std'] = np.std(values)
                aggregates[f'{metric}_min'] = np.min(values)
                aggregates[f'{metric}_max'] = np.max(values)
        
        return aggregates
    
    def _calculate_metric_statistics(
        self, 
        eval_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each metric across all evaluations."""
        if not eval_data:
            return {}
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(eval_data)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            stats = {}
            for col in numeric_columns:
                if col not in ['id', 'step']:
                    stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'median': df[col].median()
                    }
        else:
            # Manual calculation without pandas
            stats = {}
            # Get all numeric metrics
            numeric_metrics = set()
            for row in eval_data:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key not in ['id', 'step']:
                        numeric_metrics.add(key)
            
            for metric in numeric_metrics:
                values = [row.get(metric) for row in eval_data if row.get(metric) is not None]
                if values:
                    stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
        
        return stats
    
    def _generate_performance_insights(
        self, 
        eval_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate performance insights from evaluation data."""
        if not eval_data:
            return {}
        
        df = pd.DataFrame(eval_data)
        insights = {}
        
        # Identify strengths and weaknesses
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        metric_means = {}
        
        for col in numeric_columns:
            if col not in ['id', 'step']:
                metric_means[col] = df[col].mean()
        
        if metric_means:
            # Find best and worst performing metrics
            best_metric = max(metric_means.items(), key=lambda x: x[1])
            worst_metric = min(metric_means.items(), key=lambda x: x[1])
            
            insights['strongest_capability'] = {
                'metric': best_metric[0],
                'score': best_metric[1]
            }
            insights['weakest_capability'] = {
                'metric': worst_metric[0],
                'score': worst_metric[1]
            }
            
            # Identify metrics needing improvement (below 0.7)
            needs_improvement = [
                metric for metric, score in metric_means.items() 
                if score < 0.7
            ]
            insights['needs_improvement'] = needs_improvement
            
            # Calculate overall performance score
            insights['overall_score'] = np.mean(list(metric_means.values()))
        
        return insights
    
    def _filter_by_categories(
        self, 
        eval_data: List[Dict[str, Any]], 
        categories: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter evaluation data by metric categories."""
        category_metrics = {
            'entropy': ['character_entropy', 'response_entropy', 'input_entropy'],
            'capabilities': ['task_completion', 'factual_accuracy', 'reasoning_quality'],
            'alignment': ['instruction_following', 'helpfulness', 'safety'],
            'information_theory': ['information_gain', 'empowerment']
        }
        
        # Get all metrics for requested categories
        relevant_metrics = set()
        for category in categories:
            if category in category_metrics:
                relevant_metrics.update(category_metrics[category])
        
        # Filter data to include only relevant metrics
        filtered_data = []
        for row in eval_data:
            filtered_row = {k: v for k, v in row.items() 
                          if k in relevant_metrics or k in ['session_id', 'step', 'timestamp']}
            filtered_data.append(filtered_row)
        
        return filtered_data
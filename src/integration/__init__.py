"""
Integration Module - Orchestrates observation and evaluation.

This module provides the glue layer that combines observer and evaluator
functionality into cohesive analysis pipelines.
"""

from .pipelines import EvaluationPipeline, ObservationPipeline
from .reports import ReportGenerator

__all__ = ['EvaluationPipeline', 'ObservationPipeline', 'ReportGenerator']
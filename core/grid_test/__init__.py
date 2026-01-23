"""
Grid search and comparison module.

Provides grid search functionality and result comparison/visualization.
"""
from .reporter import ComparisonReporter
from .grid_search import generate_grid_configs

__all__ = ['ComparisonReporter', 'generate_grid_configs']

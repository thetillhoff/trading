"""
Grid search configuration generation.

Extracted from config.py to separate grid search logic from configuration.
"""
from ..signals.config import generate_grid_configs

__all__ = ['generate_grid_configs']

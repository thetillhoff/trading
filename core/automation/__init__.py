"""
Automation module for long-running trading service.

Handles scheduling, signal analysis, and automated order placement.
"""
from .scheduler import Scheduler
from .trader import AutomatedTrader
from .state import StateManager

__all__ = ["Scheduler", "AutomatedTrader", "StateManager"]

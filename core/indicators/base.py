"""
Base indicator interface.

All indicators should follow this pattern:
1. Calculate values from price data
2. Provide values that can be used for signal generation
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Indicator(ABC):
    """
    Base class for all indicators.
    
    Indicators calculate values from price data that can be used
    for signal generation. They do not generate signals directly.
    """
    
    @abstractmethod
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate indicator values from price data.
        
        Args:
            prices: Price series with datetime index
        
        Returns:
            Series with indicator values (same index as prices)
        """
        pass
    
    @abstractmethod
    def get_value_at(self, prices: pd.Series, timestamp: pd.Timestamp) -> Optional[float]:
        """
        Get indicator value at a specific timestamp.
        
        Args:
            prices: Price series (must include data before timestamp)
            timestamp: Timestamp to get value for
        
        Returns:
            Indicator value at timestamp, or None if insufficient data
        """
        pass

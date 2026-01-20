"""
Data processing module for aggregating and resampling time series data.

Provides functionality for different granularities and aggregation methods.
"""
import pandas as pd
from typing import Literal, Optional
from enum import Enum


class Granularity(Enum):
    """Supported time granularities."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    YEARLY = "Y"


class AggregationMethod(Enum):
    """Supported aggregation methods."""
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    SUM = "sum"
    FIRST = "first"
    LAST = "last"


class DataProcessor:
    """Processes and aggregates time series data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the data processor.
        
        Args:
            df: DataFrame with datetime index
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        self.df = df.copy()
    
    def filter_date_range(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            start_date: Start date (inclusive) in format 'YYYY-MM-DD'
            end_date: End date (inclusive) in format 'YYYY-MM-DD'
        
        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df.index >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df.index <= end]
        
        return df
    
    def resample(
        self, 
        granularity: Granularity,
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Resample data to specified granularity with aggregation.
        
        Args:
            granularity: Time granularity (DAILY, WEEKLY, MONTHLY, YEARLY)
            aggregation: Aggregation method (MEAN, MAX, MIN, etc.)
            column: Specific column to aggregate (None for all numeric columns)
        
        Returns:
            Resampled DataFrame
        """
        df = self.df.copy()
        
        # Get aggregation function
        agg_func = getattr(pd.Series, aggregation.value)
        
        # Resample
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            resampled = df[column].resample(granularity.value).apply(agg_func).to_frame()
        else:
            # Apply aggregation to all numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            resampled = df[numeric_cols].resample(granularity.value).apply(agg_func)
        
        return resampled
    
    def get_column(self, column: str) -> pd.Series:
        """
        Get a specific column from the data.
        
        Args:
            column: Column name
        
        Returns:
            Series with the column data
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found. Available columns: {list(self.df.columns)}")
        return self.df[column]
    
    def get_available_columns(self) -> list:
        """Get list of available columns."""
        return list(self.df.columns)

"""
Analyzes Elliott Wave patterns across different time granularities.

Supports yearly, quarterly, monthly, weekly, and daily analysis.
"""
import pandas as pd
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path for imports
# djia directory is mounted at /app/djia in Docker
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    # Fallback for local execution
    djia_dir = Path(__file__).parent.parent / 'djia'
sys.path.insert(0, str(djia_dir))
from elliott_wave_detector import ElliottWaveDetector, Wave, WaveType, WaveLabel


class Granularity(Enum):
    """Supported time granularities for analysis."""
    YEARLY = "Y"
    QUARTERLY = "Q"
    MONTHLY = "M"
    WEEKLY = "W"
    DAILY = "D"


@dataclass
class GranularityAnalysis:
    """Results of Elliott Wave analysis for a specific granularity."""
    granularity: Granularity
    data_points: int
    price_range: float
    waves_detected: int
    complete_patterns: int
    avg_confidence: float
    wave_size_distribution: Dict[str, float]  # min, max, mean, median
    recommended_min_confidence: float
    recommended_min_wave_size_ratio: float


class GranularityAnalyzer:
    """Analyzes Elliott Wave patterns across different time granularities."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.detector = ElliottWaveDetector()
    
    def analyze_granularity(
        self,
        data: pd.Series,
        granularity: Granularity,
        resample_method: str = 'last'
    ) -> GranularityAnalysis:
        """
        Analyze Elliott Wave patterns for a specific granularity.
        
        Args:
            data: Time series data with datetime index
            granularity: Time granularity to analyze
            resample_method: Method for resampling ('last', 'first', 'mean', etc.)
            
        Returns:
            Analysis results for this granularity
        """
        # Resample data if needed
        if granularity != Granularity.DAILY:
            resampled = self._resample_data(data, granularity, resample_method)
        else:
            resampled = data
        
        if len(resampled) < 10:
            # Not enough data for meaningful analysis
            return GranularityAnalysis(
                granularity=granularity,
                data_points=len(resampled),
                price_range=0.0,
                waves_detected=0,
                complete_patterns=0,
                avg_confidence=0.0,
                wave_size_distribution={},
                recommended_min_confidence=0.0,
                recommended_min_wave_size_ratio=0.0
            )
        
        # Detect waves with no filtering first
        waves = self.detector.detect_waves(
            resampled,
            min_confidence=0.0,
            min_wave_size_ratio=0.0,
            only_complete_patterns=False
        )
        
        # Calculate statistics
        price_range = resampled.max() - resampled.min()
        complete_patterns = self._count_complete_patterns(waves)
        
        if waves:
            avg_confidence = sum(w.confidence for w in waves) / len(waves)
            wave_sizes = [abs(w.end_price - w.start_price) for w in waves]
            wave_size_ratios = [size / price_range for size in wave_sizes if price_range > 0]
            
            wave_size_distribution = {
                'min': min(wave_size_ratios) if wave_size_ratios else 0.0,
                'max': max(wave_size_ratios) if wave_size_ratios else 0.0,
                'mean': sum(wave_size_ratios) / len(wave_size_ratios) if wave_size_ratios else 0.0,
                'median': sorted(wave_size_ratios)[len(wave_size_ratios) // 2] if wave_size_ratios else 0.0
            }
            
            # Recommend filter values
            recommended_min_confidence = max(0.5, avg_confidence - 0.1)
            recommended_min_wave_size_ratio = max(0.02, wave_size_distribution['median'] * 0.5)
        else:
            avg_confidence = 0.0
            wave_size_distribution = {}
            recommended_min_confidence = 0.6
            recommended_min_wave_size_ratio = 0.05
        
        return GranularityAnalysis(
            granularity=granularity,
            data_points=len(resampled),
            price_range=price_range,
            waves_detected=len(waves),
            complete_patterns=complete_patterns,
            avg_confidence=avg_confidence,
            wave_size_distribution=wave_size_distribution,
            recommended_min_confidence=recommended_min_confidence,
            recommended_min_wave_size_ratio=recommended_min_wave_size_ratio
        )
    
    def analyze_all_granularities(
        self,
        data: pd.Series,
        resample_method: str = 'last'
    ) -> Dict[Granularity, GranularityAnalysis]:
        """
        Analyze Elliott Wave patterns across all supported granularities.
        
        Args:
            data: Time series data with datetime index
            resample_method: Method for resampling
            
        Returns:
            Dictionary mapping granularities to their analysis results
        """
        results = {}
        
        for granularity in Granularity:
            try:
                analysis = self.analyze_granularity(data, granularity, resample_method)
                results[granularity] = analysis
            except Exception as e:
                print(f"Warning: Failed to analyze {granularity.value}: {e}")
                continue
        
        return results
    
    def _resample_data(
        self,
        data: pd.Series,
        granularity: Granularity,
        method: str = 'last'
    ) -> pd.Series:
        """Resample data to specified granularity."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        resampled = data.resample(granularity.value)
        
        if method == 'last':
            return resampled.last()
        elif method == 'first':
            return resampled.first()
        elif method == 'mean':
            return resampled.mean()
        elif method == 'median':
            return resampled.median()
        else:
            return resampled.last()
    
    def _count_complete_patterns(self, waves: List[Wave]) -> int:
        """Count complete wave patterns (5-wave impulse or 3-wave corrective)."""
        if not waves:
            return 0
        
        # Group waves by pattern
        impulse_labels = {WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                         WaveLabel.WAVE_4, WaveLabel.WAVE_5}
        corrective_labels = {WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C}
        
        # Find complete impulse patterns
        impulse_waves = [w for w in waves if w.wave_type == WaveType.IMPULSE]
        impulse_labels_found = {w.label for w in impulse_waves}
        
        # Find complete corrective patterns
        corrective_waves = [w for w in waves if w.wave_type == WaveType.CORRECTIVE]
        corrective_labels_found = {w.label for w in corrective_waves}
        
        complete_count = 0
        
        # Count complete impulse patterns
        if impulse_labels_found == impulse_labels:
            complete_count += 1
        
        # Count complete corrective patterns
        if corrective_labels_found == corrective_labels:
            complete_count += 1
        
        return complete_count

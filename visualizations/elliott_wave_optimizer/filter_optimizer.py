"""
Optimizes Elliott Wave filter parameters based on data analysis.

Finds optimal values for min_confidence and min_wave_size_ratio
by analyzing wave patterns across different granularities.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
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
from elliott_wave_detector import ElliottWaveDetector, WaveType

# Use absolute import for direct script execution
from granularity_analyzer import GranularityAnalyzer, Granularity, GranularityAnalysis


@dataclass
class FilterRecommendation:
    """Recommended filter values for Elliott Wave detection."""
    min_confidence: float
    min_wave_size_ratio: float
    only_complete_patterns: bool
    reasoning: str
    granularity_used: Granularity
    expected_wave_count: int


class FilterOptimizer:
    """Optimizes Elliott Wave filter parameters."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.analyzer = GranularityAnalyzer()
    
    def optimize_filters(
        self,
        data: pd.Series,
        target_wave_count: Optional[int] = None,
        preferred_granularity: Optional[Granularity] = None
    ) -> FilterRecommendation:
        """
        Find optimal filter parameters for Elliott Wave detection.
        
        Args:
            data: Time series data with datetime index
            target_wave_count: Desired number of waves to display (None for auto)
            preferred_granularity: Preferred granularity for analysis (None for auto)
            
        Returns:
            Recommended filter values
        """
        # Analyze all granularities
        analyses = self.analyzer.analyze_all_granularities(data)
        
        if not analyses:
            # Fallback to default values
            return FilterRecommendation(
                min_confidence=0.6,
                min_wave_size_ratio=0.05,
                only_complete_patterns=False,
                reasoning="No data available for analysis, using defaults",
                granularity_used=Granularity.DAILY,
                expected_wave_count=0
            )
        
        # Select best granularity
        if preferred_granularity and preferred_granularity in analyses:
            selected_granularity = preferred_granularity
        else:
            selected_granularity = self._select_best_granularity(analyses)
        
        analysis = analyses[selected_granularity]
        
        # Determine optimal filters
        if target_wave_count is None:
            # Auto-select based on data characteristics
            target_wave_count = self._estimate_optimal_wave_count(analysis)
        
        recommendation = self._calculate_optimal_filters(
            data,
            analysis,
            target_wave_count,
            selected_granularity
        )
        
        return recommendation
    
    def _select_best_granularity(
        self,
        analyses: Dict[Granularity, GranularityAnalysis]
    ) -> Granularity:
        """
        Select the best granularity for analysis.
        
        Prefers granularities with:
        - Sufficient data points (10+)
        - Reasonable number of waves detected
        - Good confidence scores
        """
        # Score each granularity
        scores = {}
        
        for granularity, analysis in analyses.items():
            if analysis.data_points < 10:
                continue
            
            score = 0.0
            
            # Prefer granularities with more data points (up to a point)
            score += min(analysis.data_points / 100, 1.0) * 0.3
            
            # Prefer granularities with detected waves
            if analysis.waves_detected > 0:
                score += min(analysis.waves_detected / 20, 1.0) * 0.4
            
            # Prefer higher confidence
            score += analysis.avg_confidence * 0.3
            
            scores[granularity] = score
        
        if not scores:
            # Fallback to daily
            return Granularity.DAILY
        
        # Return granularity with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _estimate_optimal_wave_count(
        self,
        analysis: GranularityAnalysis
    ) -> int:
        """
        Estimate optimal number of waves to display.
        
        Based on data points and typical wave patterns.
        """
        if analysis.data_points < 50:
            return max(1, analysis.waves_detected // 3)
        elif analysis.data_points < 200:
            return max(2, analysis.waves_detected // 4)
        elif analysis.data_points < 1000:
            return max(3, analysis.waves_detected // 5)
        else:
            # For large datasets, show more waves but still filtered
            return max(5, analysis.waves_detected // 6)
    
    def _calculate_optimal_filters(
        self,
        data: pd.Series,
        analysis: GranularityAnalysis,
        target_wave_count: int,
        granularity: Granularity
    ) -> FilterRecommendation:
        """
        Calculate optimal filter values to achieve target wave count.
        
        Uses binary search approach to find filters that yield
        approximately the target number of waves.
        """
        # Start with analysis recommendations
        min_confidence = analysis.recommended_min_confidence
        min_wave_size_ratio = analysis.recommended_min_wave_size_ratio
        
        # If we have complete patterns, consider using that option
        only_complete = analysis.complete_patterns > 0 and target_wave_count <= 5
        
        # Try to refine filters to match target count
        if analysis.waves_detected > target_wave_count:
            # Need to filter more
            min_confidence = min(0.9, min_confidence + 0.1)
            min_wave_size_ratio = min(0.2, min_wave_size_ratio * 1.5)
        
        # Build reasoning
        reasoning = (
            f"Based on {granularity.value} analysis: "
            f"{analysis.waves_detected} waves detected, "
            f"avg confidence {analysis.avg_confidence:.2f}, "
            f"recommended filters to show ~{target_wave_count} waves"
        )
        
        return FilterRecommendation(
            min_confidence=min_confidence,
            min_wave_size_ratio=min_wave_size_ratio,
            only_complete_patterns=only_complete,
            reasoning=reasoning,
            granularity_used=granularity,
            expected_wave_count=target_wave_count
        )
    
    def analyze_filter_impact(
        self,
        data: pd.Series,
        min_confidence: float,
        min_wave_size_ratio: float,
        only_complete_patterns: bool = False
    ) -> Dict[str, any]:
        """
        Analyze the impact of specific filter values.
        
        Args:
            data: Time series data
            min_confidence: Confidence threshold
            min_wave_size_ratio: Wave size ratio threshold
            only_complete_patterns: Whether to only show complete patterns
            
        Returns:
            Dictionary with analysis results
        """
        detector = ElliottWaveDetector()
        waves = detector.detect_waves(
            data,
            min_confidence=min_confidence,
            min_wave_size_ratio=min_wave_size_ratio,
            only_complete_patterns=only_complete_patterns
        )
        
        return {
            'wave_count': len(waves),
            'impulse_waves': len([w for w in waves if w.wave_type == WaveType.IMPULSE]),
            'corrective_waves': len([w for w in waves if w.wave_type == WaveType.CORRECTIVE]),
            'avg_confidence': sum(w.confidence for w in waves) / len(waves) if waves else 0.0,
            'complete_patterns': self.analyzer._count_complete_patterns(waves)
        }

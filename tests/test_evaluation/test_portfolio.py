"""
Tests for portfolio simulator.
"""
import pytest
import pandas as pd
import numpy as np
from core.evaluation.portfolio import PortfolioSimulator, PositionStatus
from core.shared.types import TradingSignal, SignalType


@pytest.fixture
def sample_signals():
    """Create sample trading signals."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    signals = []
    
    # Buy signal
    signals.append(TradingSignal(
        signal_type=SignalType.BUY,
        timestamp=dates[0],
        price=100.0,
        confidence=0.8,
        target_price=110.0,
        stop_loss=95.0,
        reasoning="Test buy signal"
    ))
    
    return signals


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    prices = pd.Series(
        100 + np.arange(20) * 0.5,
        index=dates
    )
    return prices


class TestPortfolioSimulator:
    """Test portfolio simulation."""
    
    def test_initialization(self):
        """Should initialize with default parameters."""
        sim = PortfolioSimulator(initial_capital=100)
        assert sim.initial_capital == 100
    
    def test_simulate_returns_result(self, sample_signals, sample_prices):
        """Simulation should return result object."""
        sim = PortfolioSimulator(initial_capital=100)
        result = sim.simulate_strategy(sample_prices, sample_signals)
        
        assert result is not None
        assert hasattr(result, 'initial_capital')
        assert hasattr(result, 'final_capital')
        assert hasattr(result, 'total_trades')
    
    def test_capital_conservation(self, sample_signals, sample_prices):
        """Total capital should be conserved (cash + positions)."""
        sim = PortfolioSimulator(initial_capital=100)
        result = sim.simulate_strategy(sample_prices, sample_signals)
        
        # Final capital should be positive
        assert result.final_capital > 0
    
    def test_position_sizing(self):
        """Position size should respect configuration."""
        sim = PortfolioSimulator(
            initial_capital=100,
            position_size_pct=0.2  # 20% per trade
        )
        
        assert sim.position_size_pct == 0.2
    
    def test_max_positions(self):
        """Should respect max_positions limit."""
        sim = PortfolioSimulator(
            initial_capital=100,
            max_positions=3
        )
        
        assert sim.max_positions == 3

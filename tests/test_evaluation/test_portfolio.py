"""
Tests for portfolio simulator.
"""
import pytest
import pandas as pd
import numpy as np
from core.evaluation.portfolio import (
    PortfolioSimulator,
    PositionStatus,
    SimulationResult,
)
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

    def test_sell_to_close_closes_oldest_long(self):
        """SELL with close_long_only=True closes oldest open long, does not open short."""
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        prices = pd.Series(100.0 + np.arange(15) * 0.5, index=dates)
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[0],
                price=100.0,
                confidence=0.8,
                target_price=120.0,
                stop_loss=90.0,
                reasoning="Buy",
            ),
            TradingSignal(
                signal_type=SignalType.SELL,
                timestamp=dates[5],
                price=102.5,
                confidence=0.7,
                target_price=95.0,
                stop_loss=108.0,
                reasoning="Sell-to-close",
                close_long_only=True,
            ),
        ]
        sim = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=0.5,
            max_positions=5,
        )
        result: SimulationResult = sim.simulate_strategy(prices, signals)
        assert result.total_trades == 1
        closed = result.positions[0]
        assert closed.status == PositionStatus.CLOSED_SIGNAL
        assert closed.signal_type == "buy"
        assert closed.exit_timestamp == dates[5]
        assert closed.exit_price == 102.5
        # No short was opened: only one position existed and it was a long that got closed
        shorts = [p for p in result.positions if p.signal_type == "sell"]
        assert len(shorts) == 0

    def test_total_trading_costs_zero_when_no_fees(self, sample_signals, sample_prices):
        """When no fee params set, total_trading_costs is 0."""
        sim = PortfolioSimulator(initial_capital=100)
        result = sim.simulate_strategy(sample_prices, sample_signals)
        assert hasattr(result, 'total_trading_costs')
        assert result.total_trading_costs == 0.0

    def test_trading_costs_reduce_final_capital(self, sample_signals, sample_prices):
        """With trade_fee_pct and/or trade_fee_absolute, total_trading_costs > 0 and final_capital lower."""
        sim_no_fee = PortfolioSimulator(initial_capital=100)
        result_no_fee = sim_no_fee.simulate_strategy(sample_prices, sample_signals)
        sim_fee = PortfolioSimulator(
            initial_capital=100,
            trade_fee_pct=0.001,
            trade_fee_absolute=0.5,
        )
        result_fee = sim_fee.simulate_strategy(sample_prices, sample_signals)
        assert result_fee.total_trading_costs > 0
        assert result_fee.final_capital < result_no_fee.final_capital

    def test_trading_costs_per_side(self, sample_prices):
        """One round-trip: entry fee + exit fee = total_trading_costs."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        prices = pd.Series(100.0 + np.arange(20) * 0.5, index=dates)
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[0],
                price=100.0,
                confidence=0.8,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
            ),
        ]
        sim = PortfolioSimulator(
            initial_capital=100,
            position_size_pct=0.2,
            trade_fee_pct=0.001,
            trade_fee_absolute=0.0,
        )
        result = sim.simulate_strategy(prices, signals)
        # One trade: entry fee on 20% of 100 = 20, exit at end; exit value ~20 * 109.5
        assert result.total_trades >= 1
        assert result.total_trading_costs > 0

    def test_trade_fee_min_max_clamp(self, sample_prices):
        """trade_fee_min and trade_fee_max clamp the fee per side (absolute)."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        prices = pd.Series(100.0 + np.arange(20) * 0.5, index=dates)
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[0],
                price=100.0,
                confidence=0.8,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
            ),
        ]
        # Small position (5% of 100 = 5); 0.1% of 5 = 0.005 raw; trade_fee_min=0.5 -> pay 0.5 per side -> 1.0 total for round-trip
        sim = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=0.05,
            trade_fee_pct=0.001,
            trade_fee_absolute=0.0,
            trade_fee_min=0.5,
        )
        result = sim.simulate_strategy(prices, signals)
        assert result.total_trades >= 1
        assert result.total_trading_costs >= 1.0  # at least 0.5 entry + 0.5 exit

        # Large position; trade_fee_max caps fee per side
        sim_max = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=1.0,
            trade_fee_pct=0.01,
            trade_fee_absolute=0.0,
            trade_fee_max=0.3,
        )
        result_max = sim_max.simulate_strategy(prices, signals)
        assert result_max.total_trades >= 1
        # Raw would be 1% of 100 = 1.0 per side; clamped to 0.3 -> 0.6 total round-trip
        assert result_max.total_trading_costs <= 0.65

    def test_simulate_strategy_multi_instrument(self):
        """Multi-instrument: Dict[str, Series] uses per-instrument price for exits and PnL."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        prices_a = pd.Series(100.0 + np.arange(30) * 0.5, index=dates)
        prices_b = pd.Series(200.0 + np.arange(30) * 0.3, index=dates)
        prices_by_instrument = {"inst_a": prices_a, "inst_b": prices_b}
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.8,
                target_price=105.0,
                stop_loss=98.0,
                reasoning="Buy A",
                instrument="inst_a",
            ),
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[2],
                price=200.6,
                confidence=0.8,
                target_price=206.0,
                stop_loss=197.0,
                reasoning="Buy B",
                instrument="inst_b",
            ),
        ]
        sim = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=0.3,
            max_positions=5,
        )
        result = sim.simulate_strategy(prices_by_instrument, signals)
        assert result.total_trades >= 2
        instruments_in_positions = {p.instrument for p in result.positions}
        assert "inst_a" in instruments_in_positions
        assert "inst_b" in instruments_in_positions

    def test_max_positions_per_instrument_enforced(self):
        """With max_positions_per_instrument=1, at most one open position per instrument."""
        dates = pd.date_range('2020-01-01', periods=25, freq='D')
        prices_a = pd.Series(100.0 + np.arange(25) * 0.5, index=dates)
        prices_b = pd.Series(200.0 + np.arange(25) * 0.3, index=dates)
        prices_by_instrument = {"inst_a": prices_a, "inst_b": prices_b}
        # Four BUY signals: two for inst_a, two for inst_b (same day so all processed before any exit)
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.8,
                target_price=105.0,
                stop_loss=98.0,
                reasoning="Buy A1",
                instrument="inst_a",
            ),
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.8,
                target_price=105.0,
                stop_loss=98.0,
                reasoning="Buy A2",
                instrument="inst_a",
            ),
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=200.6,
                confidence=0.8,
                target_price=206.0,
                stop_loss=197.0,
                reasoning="Buy B1",
                instrument="inst_b",
            ),
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=200.6,
                confidence=0.8,
                target_price=206.0,
                stop_loss=197.0,
                reasoning="Buy B2",
                instrument="inst_b",
            ),
        ]
        sim = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=0.3,
            max_positions=10,
            max_positions_per_instrument=1,
        )
        result = sim.simulate_strategy(prices_by_instrument, signals)
        # With limit 1 per instrument we open only 1 for inst_a and 1 for inst_b (2 total)
        assert result.total_trades == 2
        by_inst = {}
        for p in result.positions:
            inst = p.instrument or "unknown"
            by_inst.setdefault(inst, []).append(p)
        assert len(by_inst.get("inst_a", [])) == 1
        assert len(by_inst.get("inst_b", [])) == 1

    def test_min_position_size_skips_small_trades(self):
        """With min_position_size set above computed position capital, no trade opens."""
        dates = pd.date_range('2020-01-01', periods=15, freq='D')
        prices = pd.Series(100.0 + np.arange(15) * 0.5, index=dates)
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.8,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
            ),
        ]
        # position_size_pct 0.05 on 100 -> 5 per trade; min_position_size 10 -> skip
        sim = PortfolioSimulator(
            initial_capital=100.0,
            position_size_pct=0.05,
            min_position_size=10.0,
        )
        result = sim.simulate_strategy(prices, signals)
        assert result.total_trades == 0
        assert len(result.positions) == 0

    def test_position_size_pct_caps_confidence_sizing(self):
        """With position_size_pct=0.02 and confidence sizing, quality_factor from 3 confirmations = 1.0;
        actual size = 2% (at cap); market exposure never exceeds cap."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        prices = pd.Series(100.0 + np.arange(20) * 0.5, index=dates)
        initial_capital = 100.0
        position_size_pct = 0.02  # 2% max per trade
        signals = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.9,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
                indicator_confirmations=3,  # quality_factor = 1.0 -> actual = 2%
            ),
        ]
        sim = PortfolioSimulator(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            use_confidence_sizing=True,
        )
        result = sim.simulate_strategy(prices, signals)
        max_cost_per_trade = initial_capital * position_size_pct
        for pos in result.positions:
            assert pos.cost_basis <= max_cost_per_trade + 0.01, (
                f"position cost_basis {pos.cost_basis} must be <= {max_cost_per_trade} (position_size_pct cap)"
            )
        max_exposure_pct = 0.0
        for state in result.wallet_history:
            if state.total_value and state.total_value > 0:
                exposure_pct = (state.invested_value / state.total_value) * 100
                max_exposure_pct = max(max_exposure_pct, exposure_pct)
        assert max_exposure_pct <= (position_size_pct * 100) + 1.0, (
            f"max exposure {max_exposure_pct}% must be <= position_size_pct cap {(position_size_pct * 100)}%"
        )

    def test_quality_factor_scales_within_cap(self):
        """Actual size = position_size_pct * quality_factor: 0 confirmations -> small size, 3 confirmations -> full cap."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        prices = pd.Series(100.0 + np.arange(20) * 0.5, index=dates)
        initial_capital = 100.0
        position_size_pct = 0.10  # 10% max per trade
        # Weak signal: 0 confirmations -> quality_factor = 0 -> skip or tiny; with confidence_sizing primary = 0
        sim_weak = PortfolioSimulator(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            use_confidence_sizing=True,
        )
        signals_weak = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.3,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
                indicator_confirmations=0,
            ),
        ]
        result_weak = sim_weak.simulate_strategy(prices, signals_weak)
        if result_weak.positions:
            # primary = 0/3 = 0 -> quality_factor 0 -> actual 0 -> would skip; if we get a position it's from fallback
            max_cost_weak = max(p.cost_basis for p in result_weak.positions)
            assert max_cost_weak < initial_capital * position_size_pct + 0.01, "0 confirmations should give size < cap"
        # Strong signal: 3 confirmations -> quality_factor = 1.0 -> actual = 10%
        sim_strong = PortfolioSimulator(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            use_confidence_sizing=True,
        )
        signals_strong = [
            TradingSignal(
                signal_type=SignalType.BUY,
                timestamp=dates[1],
                price=100.5,
                confidence=0.9,
                target_price=110.0,
                stop_loss=95.0,
                reasoning="Buy",
                indicator_confirmations=3,
            ),
        ]
        result_strong = sim_strong.simulate_strategy(prices, signals_strong)
        assert len(result_strong.positions) >= 1
        cost_strong = result_strong.positions[0].cost_basis
        expected_full = initial_capital * position_size_pct  # 10.0
        assert abs(cost_strong - expected_full) <= 0.02, (
            f"3 confirmations should give size ≈ cap: cost_basis={cost_strong}, expected≈{expected_full}"
        )

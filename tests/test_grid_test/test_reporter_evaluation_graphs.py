"""Smoke tests for evaluation graph reporter methods (pnl vs duration, confidence/risk, gain/trades per instrument, indicator best/worst)."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.evaluation.portfolio import Position, PositionStatus, SimulationResult, WalletState
from core.evaluation.walk_forward import WalkForwardResult
from core.grid_test.reporter import (
    ComparisonReporter,
    compute_alpha_over_time_series,
    _daily_rate_from_pa,
    _is_new_month,
    trades_to_dataframe,
)


def _minimal_result_with_positions(positions, wallet_history=None):
    """Build a minimal WalkForwardResult with given positions."""
    if wallet_history is None:
        wallet_history = [
            WalletState(
                timestamp=pd.Timestamp("2020-06-01"),
                cash=80.0,
                invested_value=20.0,
                total_value=100.0,
                return_pct=0.0,
            )
        ]
    config = SimpleNamespace(
        name="test",
        instruments=["djia"],
        interest_rate_pa=0.02,
    )
    sim = SimulationResult(
        initial_capital=100.0,
        final_capital=100.0,
        total_return_pct=0.0,
        total_trades=len([p for p in positions if p.exit_timestamp]),
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        wallet_history=wallet_history,
        positions=positions,
        max_drawdown_pct=0.0,
        avg_position_size=20.0,
        avg_days_held=10.0,
    )
    return WalkForwardResult(
        config=config,
        simulation=sim,
        evaluation_start_date=pd.Timestamp("2020-01-01"),
        evaluation_end_date=pd.Timestamp("2020-12-31"),
        lookback_days=365,
        step_days=1,
        buy_and_hold_gain=5.0,
        exposure_adjusted_market=2.0,
        outperformance=0.0,
        hybrid_return=0.0,
        active_alpha=0.0,
    )


def test_trades_to_dataframe_includes_instrument():
    """trades_to_dataframe output includes instrument column."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
        instrument="djia",
    )
    result = _minimal_result_with_positions([pos])
    df = trades_to_dataframe(result)
    assert not df.empty
    assert "instrument" in df.columns
    assert df["instrument"].iloc[0] == "djia"


def test_trades_to_dataframe_instrument_empty_when_none():
    """When position.instrument is None, column is empty string."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
        instrument=None,
    )
    result = _minimal_result_with_positions([pos])
    df = trades_to_dataframe(result)
    assert "instrument" in df.columns
    assert df["instrument"].iloc[0] == ""


def test_pnl_vs_duration_returns_path_when_positions():
    """generate_pnl_vs_duration_scatter returns non-empty path when result has closed positions."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
    )
    result = _minimal_result_with_positions([pos])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_pnl_vs_duration_scatter(result, filename="test_scatter.png")
        assert path
        assert path.endswith("test_scatter.png")


def test_pnl_vs_duration_returns_empty_when_no_closed():
    """generate_pnl_vs_duration_scatter returns empty string when no closed positions."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=None,
        status=PositionStatus.OPEN,
        pnl=0.0,
    )
    result = _minimal_result_with_positions([pos])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_pnl_vs_duration_scatter(result)
        assert path == ""


def test_confidence_risk_vs_pnl_returns_path():
    """generate_confidence_risk_vs_pnl_scatter returns path when closed positions."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
        certainty=0.7,
        risk_reward_ratio=2.0,
    )
    result = _minimal_result_with_positions([pos])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_confidence_risk_vs_pnl_scatter(result, filename="test_conf.png")
        assert path
        assert "test_conf.png" in path


def test_gain_per_instrument_returns_path():
    """generate_gain_per_instrument returns path when closed positions with cost_basis."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
        instrument="djia",
    )
    result = _minimal_result_with_positions([pos])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_gain_per_instrument(result, filename="test_gain.png")
        assert path
        assert "test_gain.png" in path


def test_trades_per_instrument_returns_path():
    """generate_trades_per_instrument returns path when closed positions."""
    pos = Position(
        entry_timestamp=pd.Timestamp("2020-06-01"),
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=pd.Timestamp("2020-06-15"),
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
    )
    result = _minimal_result_with_positions([pos])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_trades_per_instrument(result, filename="test_trades.png")
        assert path
        assert "test_trades.png" in path


def test_indicator_best_worst_returns_path_when_enough_trades():
    """generate_indicator_best_worst_overview returns path when enough closed positions."""
    positions = []
    for i in range(6):
        pnl = 2.0 if i < 3 else -1.0
        pos = Position(
            entry_timestamp=pd.Timestamp("2020-06-01") + pd.Timedelta(days=i),
            entry_price=100.0,
            shares=0.2,
            cost_basis=20.0,
            target_price=110.0,
            stop_loss=95.0,
            signal_type="buy",
            exit_timestamp=pd.Timestamp("2020-06-15") + pd.Timedelta(days=i),
            exit_price=105.0 + pnl,
            status=PositionStatus.CLOSED_TARGET,
            pnl=pnl,
            rsi_value=50.0 + i,
            indicator_confirmations=2,
            certainty=0.5 + i * 0.05,
        )
        positions.append(pos)
    result = _minimal_result_with_positions(positions)
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_indicator_best_worst_overview(result, filename="test_ind.png")
        assert path
        assert "test_ind.png" in path
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_indicator_best_worst_returns_empty_when_no_closed():
    """generate_indicator_best_worst_overview returns empty when no closed positions with cost_basis."""
    result = _minimal_result_with_positions([])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_indicator_best_worst_overview(result)
        assert path == ""


def test_performance_timings_chart_returns_path_when_timings_present():
    """generate_performance_timings_chart writes chart and returns path when result has performance_timings."""
    result = _minimal_result_with_positions([])
    result.performance_timings = {
        "signal_detection": 10.5,
        "portfolio_simulation": 0.2,
        "indicator_rsi": 2.1,
        "indicator_ema": 1.8,
    }
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_performance_timings_chart(result, filename="perf.png")
        assert path
        assert "perf.png" in path


def test_performance_timings_chart_returns_empty_when_no_timings():
    """generate_performance_timings_chart returns empty when result has no performance_timings."""
    result = _minimal_result_with_positions([])
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_performance_timings_chart(result)
        assert path == ""


def test_grid_performance_timings_chart_returns_path_when_timings_present():
    """generate_grid_performance_timings_chart writes chart and returns path when timings dict provided."""
    timings = {
        "data_prep": 1.2,
        "signal_detection": 45.0,
        "data_load": 3.1,
        "portfolio_simulation": 2.0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_grid_performance_timings_chart(timings, filename="grid_perf.png")
        assert path
        assert "grid_perf.png" in path


def test_grid_performance_timings_chart_returns_empty_when_no_timings():
    """generate_grid_performance_timings_chart returns empty when timings dict is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_grid_performance_timings_chart({})
        assert path == ""


def test_alpha_over_time_zero_trades_cash_earns_interest_not_bh():
    """With zero trades, Cash only and Strategy should earn ~2% p.a. (interest), not track B&H."""
    # One year of daily dates (trading days)
    eval_start = pd.Timestamp("2020-01-01")
    eval_end = pd.Timestamp("2020-12-31")
    common_index = pd.date_range(start=eval_start, end=eval_end, freq="B")
    initial_capital = 100.0
    # Zero trades: wallet_history has total_value=initial_capital every day
    wallet_history = [
        WalletState(timestamp=ts, cash=initial_capital, invested_value=0.0, total_value=initial_capital, return_pct=0.0)
        for ts in common_index
    ]
    config = SimpleNamespace(name="zero_trades", instruments=["djia"], interest_rate_pa=0.02)
    sim = SimulationResult(
        initial_capital=initial_capital,
        final_capital=initial_capital,
        total_return_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        wallet_history=wallet_history,
        positions=[],
        max_drawdown_pct=0.0,
        avg_position_size=0.0,
        avg_days_held=0.0,
    )
    result = WalkForwardResult(
        config=config,
        simulation=sim,
        evaluation_start_date=eval_start,
        evaluation_end_date=eval_end,
        lookback_days=365,
        step_days=1,
        buy_and_hold_gain=10.0,
        exposure_adjusted_market=0.0,
        outperformance=0.0,
        hybrid_return=0.0,
        active_alpha=0.0,
    )
    # Price data: B&H gains 10% over the year (linear)
    price_data = pd.Series(
        np.linspace(100.0, 110.0, len(common_index)),
        index=common_index,
    )

    # Same interest logic as reporter: display balance only (month-end payout), no accrued in display
    daily_rate = _daily_rate_from_pa(0.02)
    cash_balance = initial_capital
    accrued = 0.0
    prev_date = None
    for current_date in common_index:
        if _is_new_month(current_date, prev_date):
            cash_balance += accrued
            accrued = 0.0
        days = 1 if prev_date is None else (current_date - prev_date).days
        if days > 0:
            accrued += cash_balance * (daily_rate * days)
        prev_date = current_date
    final_cash_displayed = cash_balance
    cash_only_final_return_pct = ((final_cash_displayed - initial_capital) / initial_capital) * 100

    interest_account = 0.0
    accrued = 0.0
    prev_date = None
    for date, wallet_state in zip([w.timestamp for w in wallet_history], wallet_history):
        if _is_new_month(date, prev_date):
            interest_account += accrued
            accrued = 0.0
        days = 1 if prev_date is None else (date - prev_date).days
        if days > 0 and (wallet_state.cash + interest_account) > 0:
            accrued += (wallet_state.cash + interest_account) * (daily_rate * days)
        prev_date = date
    final_strategy_displayed = wallet_history[-1].total_value + interest_account
    strategy_final_return_pct = ((final_strategy_displayed - initial_capital) / initial_capital) * 100

    bh_final_return_pct = ((price_data.iloc[-1] / price_data.iloc[0]) - 1) * 100

    assert 1.5 <= cash_only_final_return_pct <= 2.5
    assert 1.5 <= strategy_final_return_pct <= 2.5
    assert abs(strategy_final_return_pct - bh_final_return_pct) > 1.0

    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_alpha_over_time(result, price_data, filename="alpha_zero.png")
        assert path
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_alpha_over_time_single_trade_embedded_data():
    """Single trade with time before and after: embedded price series and wallet_history; strategy distinct from B&H."""
    # 30 business days: no trade days 0-9, one trade days 10-19, no trade days 20-29
    common_index = pd.date_range(start="2020-06-01", periods=30, freq="B")
    initial_capital = 100.0
    # Price: flat 100, then 100->105 during trade window, then flat 105
    prices = np.ones(30, dtype=float) * 100.0
    prices[10:20] = np.linspace(100.0, 105.0, 10)
    prices[20:] = 105.0
    price_data = pd.Series(prices, index=common_index)

    # One position: entry day 10, exit day 20; 20% of capital (20 units), entry 100, exit 105, pnl=1
    entry_ts = common_index[10]
    exit_ts = common_index[20]
    pos = Position(
        entry_timestamp=entry_ts,
        entry_price=100.0,
        shares=0.2,
        cost_basis=20.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_type="buy",
        exit_timestamp=exit_ts,
        exit_price=105.0,
        status=PositionStatus.CLOSED_TARGET,
        pnl=1.0,
        instrument="djia",
    )

    # Wallet history: before trade all cash; during trade 80 cash + mark-to-market invested; after close 101 cash
    wallet_history = []
    for i, ts in enumerate(common_index):
        if i < 10:
            cash, invested = initial_capital, 0.0
        elif i < 20:
            # Position open: 80 cash, invested = shares * current_price
            current_price = float(price_data.iloc[i])
            invested = 0.2 * current_price
            cash = 80.0
        else:
            cash, invested = 101.0, 0.0
        total = cash + invested
        ret_pct = ((total - initial_capital) / initial_capital) * 100
        wallet_history.append(
            WalletState(timestamp=ts, cash=cash, invested_value=invested, total_value=total, return_pct=ret_pct)
        )

    config = SimpleNamespace(name="single_trade", instruments=["djia"], interest_rate_pa=0.02)
    sim = SimulationResult(
        initial_capital=initial_capital,
        final_capital=101.0,
        total_return_pct=1.0,
        total_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=100.0,
        wallet_history=wallet_history,
        positions=[pos],
        max_drawdown_pct=0.0,
        avg_position_size=20.0,
        avg_days_held=10.0,
    )
    result = WalkForwardResult(
        config=config,
        simulation=sim,
        evaluation_start_date=common_index[0],
        evaluation_end_date=common_index[-1],
        lookback_days=365,
        step_days=1,
        buy_and_hold_gain=5.0,
        exposure_adjusted_market=1.0,
        outperformance=0.0,
        hybrid_return=0.0,
        active_alpha=0.0,
    )

    series = compute_alpha_over_time_series(result, price_data)
    assert series is not None, "compute_alpha_over_time_series should return data"

    assert len(series.common_index) == 30
    bh_final = float(series.bh_returns_pct.iloc[-1])
    strategy_final = float(series.strategy_returns_pct_aligned[-1])
    assert abs(bh_final - 5.0) < 0.5, "B&H final return should be ~5%"
    assert abs(strategy_final - 1.0) < 0.5, "strategy final return should be ~1%"
    assert abs(strategy_final - bh_final) > 2.0, "strategy and B&H must be distinct"

    assert series.cash_only_returns_pct[0] == 0.0
    assert series.strategy_returns_pct_aligned[0] == 0.0


def test_generate_market_exposure_returns_path_when_wallet_history():
    """generate_market_exposure returns non-empty path and creates file when result has wallet_history."""
    wallet_history = [
        WalletState(
            timestamp=pd.Timestamp("2020-06-01"),
            cash=80.0,
            invested_value=20.0,
            total_value=100.0,
            return_pct=0.0,
        ),
        WalletState(
            timestamp=pd.Timestamp("2020-06-02"),
            cash=100.0,
            invested_value=0.0,
            total_value=100.0,
            return_pct=0.0,
        ),
    ]
    result = _minimal_result_with_positions([], wallet_history=wallet_history)
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ComparisonReporter(output_dir=tmp)
        path = reporter.generate_market_exposure(result, filename="market_exposure.png")
        assert path
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

"""
Tests for trade_analysis (aggregate by signal type).
"""
import pytest
import pandas as pd
from core.evaluation.trade_analysis import (
    aggregate_positions_by_signal_type,
    aggregate_trades_dataframe_by_signal_type,
)
from core.evaluation.portfolio import Position, PositionStatus


def _pos(signal_type: str, pnl: float, cost_basis: float, status: PositionStatus = PositionStatus.CLOSED_TARGET):
    return Position(
        entry_timestamp=pd.Timestamp("2020-01-01"),
        entry_price=100.0,
        shares=1.0,
        cost_basis=cost_basis,
        target_price=110.0,
        stop_loss=90.0,
        signal_type=signal_type,
        exit_timestamp=pd.Timestamp("2020-01-02"),
        exit_price=100.0,
        status=status,
        pnl=pnl,
    )


class TestAggregatePositionsBySignalType:
    def test_empty(self):
        out = aggregate_positions_by_signal_type([])
        assert out["buy"]["count"] == 0
        assert out["sell"]["count"] == 0
        assert out["buy"]["total_pnl"] == 0.0
        assert out["sell"]["total_pnl"] == 0.0

    def test_excludes_open(self):
        pos = _pos("buy", 10.0, 50.0, status=PositionStatus.OPEN)
        pos.exit_timestamp = None
        out = aggregate_positions_by_signal_type([pos])
        assert out["buy"]["count"] == 0
        assert out["sell"]["count"] == 0

    def test_excludes_zero_cost_basis(self):
        pos = _pos("buy", 10.0, 0.0)
        out = aggregate_positions_by_signal_type([pos])
        assert out["buy"]["count"] == 0

    def test_aggregates_buy_sell(self):
        positions = [
            _pos("buy", 5.0, 50.0),
            _pos("buy", -2.0, 50.0),
            _pos("sell", 1.0, 50.0),
            _pos("sell", 1.0, 50.0),
        ]
        out = aggregate_positions_by_signal_type(positions)
        assert out["buy"]["count"] == 2
        assert out["buy"]["total_pnl"] == 3.0
        assert out["buy"]["win_rate_pct"] == 50.0
        assert out["sell"]["count"] == 2
        assert out["sell"]["total_pnl"] == 2.0
        assert out["sell"]["win_rate_pct"] == 100.0

    def test_avg_win_loss_pct(self):
        # buy: +10% and -4% -> avg_win 10, avg_loss -4
        positions = [
            _pos("buy", 5.0, 50.0),   # 10%
            _pos("buy", -2.0, 50.0),  # -4%
        ]
        out = aggregate_positions_by_signal_type(positions)
        assert out["buy"]["avg_win_pct"] == 10.0
        assert out["buy"]["avg_loss_pct"] == -4.0
        assert out["buy"]["avg_pnl_pct"] == 3.0  # (10 - 4) / 2


class TestAggregateTradesDataframeBySignalType:
    def test_empty(self):
        df = pd.DataFrame(columns=["signal_type", "pnl", "pnl_pct", "cost_basis", "status"])
        out = aggregate_trades_dataframe_by_signal_type(df)
        assert out["buy"]["count"] == 0
        assert out["sell"]["count"] == 0

    def test_filters_closed_and_positive_cost(self):
        df = pd.DataFrame([
            {"signal_type": "buy", "pnl": 5.0, "cost_basis": 50.0, "status": "open"},
            {"signal_type": "buy", "pnl": 3.0, "cost_basis": 50.0, "status": "closed_target"},
        ])
        out = aggregate_trades_dataframe_by_signal_type(df)
        assert out["buy"]["count"] == 1
        assert out["buy"]["total_pnl"] == 3.0

    def test_computes_pnl_pct_if_missing(self):
        df = pd.DataFrame([
            {"signal_type": "buy", "pnl": 5.0, "cost_basis": 50.0, "status": "closed_target"},
        ])
        out = aggregate_trades_dataframe_by_signal_type(df)
        assert out["buy"]["count"] == 1
        assert out["buy"]["avg_pnl_pct"] == 10.0  # 5/50*100
        assert out["buy"]["total_pnl_pct"] == 10.0

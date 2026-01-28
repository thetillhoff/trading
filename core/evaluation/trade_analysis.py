"""
Trade analysis helpers: aggregate by signal type (long vs short).
"""
from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from .portfolio import Position, PositionStatus


def _closed_and_valid(pos: Position) -> bool:
    return (
        pos.status != PositionStatus.OPEN
        and pos.cost_basis > 0
        and pos.exit_timestamp is not None
    )


def _metrics_for_trades(
    pnls: List[float],
    pnl_pcts: List[float],
    cost_bases: List[float],
) -> Dict[str, Any]:
    n = len(pnls)
    if n == 0:
        return {
            "count": 0,
            "win_rate_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "avg_pnl_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }
    total_pnl = sum(pnls)
    total_cost = sum(cost_bases)
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
    winners = [p for p in pnl_pcts if p > 0]
    losers = [p for p in pnl_pcts if p < 0]
    win_rate_pct = (len(winners) / n * 100) if n else 0.0
    avg_pnl_pct = sum(pnl_pcts) / n if n else 0.0
    avg_win_pct = (sum(winners) / len(winners)) if winners else 0.0
    avg_loss_pct = (sum(losers) / len(losers)) if losers else 0.0
    return {
        "count": n,
        "win_rate_pct": win_rate_pct,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }


def aggregate_positions_by_signal_type(
    positions: List[Position],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate closed positions by signal_type (buy=long, sell=short).

    Filters to positions with status != OPEN, cost_basis > 0, and exit_timestamp set.
    Returns a dict keyed by "buy" and "sell" with count, win_rate_pct, total_pnl,
    total_pnl_pct, avg_pnl_pct, avg_win_pct, avg_loss_pct.
    """
    closed = [p for p in positions if _closed_and_valid(p)]
    out: Dict[str, Dict[str, Any]] = {"buy": {}, "sell": {}}
    for side in ("buy", "sell"):
        subset = [p for p in closed if (p.signal_type or "").lower() == side]
        pnls = [p.pnl for p in subset]
        cost_bases = [p.cost_basis for p in subset]
        pnl_pcts = [
            (p.pnl / p.cost_basis * 100) if p.cost_basis > 0 else 0.0
            for p in subset
        ]
        out[side] = _metrics_for_trades(pnls, pnl_pcts, cost_bases)
    return out


def aggregate_trades_dataframe_by_signal_type(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Same metrics as aggregate_positions_by_signal_type but from a trades DataFrame.

    Expects columns: signal_type, pnl, pnl_pct, cost_basis, status.
    Filters to closed rows (status not 'open') and cost_basis > 0.
    """
    if df.empty:
        return {"buy": _metrics_for_trades([], [], []), "sell": _metrics_for_trades([], [], [])}
    closed = df[
        (df["status"].str.lower() != "open")
        & (df["cost_basis"] > 0)
    ].copy()
    if "pnl_pct" not in closed.columns:
        closed["pnl_pct"] = (
            closed["pnl"] / closed["cost_basis"] * 100
        ).fillna(0)
    out: Dict[str, Dict[str, Any]] = {"buy": {}, "sell": {}}
    for side in ("buy", "sell"):
        subset = closed[(closed["signal_type"].str.lower() == side)]
        pnls = subset["pnl"].tolist()
        pnl_pcts = subset["pnl_pct"].tolist()
        cost_bases = subset["cost_basis"].tolist()
        out[side] = _metrics_for_trades(pnls, pnl_pcts, cost_bases)
    return out

"""
Alpha-over-time series computation for reporter charts.

Computes cash-only, buy-and-hold, strategy, and alpha series used by generate_alpha_over_time.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..evaluation.walk_forward import WalkForwardResult
from .reporter_utils import _daily_rate_from_pa, _is_new_month


@dataclass
class AlphaOverTimeSeries:
    """Computed series for alpha-over-time chart (cash, B&H, strategy, alpha)."""
    common_index: pd.DatetimeIndex
    cash_only_returns_pct: List[float]
    bh_returns_pct: pd.Series
    bh_series_by_inst: Dict[str, pd.Series]
    strategy_returns_pct_aligned: np.ndarray
    alpha_dates: List
    alpha_over_time: List[float]
    interest_rate_pa: float


def compute_alpha_over_time_series(
    result: WalkForwardResult,
    price_data: pd.Series,
    price_data_by_instrument: Optional[Dict[str, pd.Series]] = None,
) -> Optional[AlphaOverTimeSeries]:
    """
    Compute series for alpha-over-time (cash only, B&H, strategy, alpha). No chart.
    Returns None on early exit (no result, no wallet_history, empty eval_data, no B&H, empty alpha).
    """
    if not result or not result.simulation.wallet_history:
        return None

    eval_start = result.evaluation_start_date
    eval_end = result.evaluation_end_date
    initial_capital = result.simulation.initial_capital
    interest_rate_pa = getattr(result.config, 'interest_rate_pa', 0.02)
    daily_rate = _daily_rate_from_pa(interest_rate_pa)

    eval_data = price_data[
        (price_data.index >= eval_start) &
        (price_data.index <= eval_end)
    ]
    if len(eval_data) == 0:
        return None
    common_index = eval_data.index

    cash_only_values = []
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
        cash_only_values.append(cash_balance)
        prev_date = current_date
    cash_only_returns_pct = [((v - initial_capital) / initial_capital) * 100 for v in cash_only_values]

    bh_series_by_inst: Dict[str, pd.Series] = {}
    if price_data_by_instrument:
        instruments_ordered = [k for k in (getattr(result.config, 'instruments', None) or []) if k in price_data_by_instrument]
        if not instruments_ordered:
            instruments_ordered = sorted(price_data_by_instrument.keys())
        for inst in instruments_ordered:
            ser = price_data_by_instrument.get(inst)
            if ser is None or len(ser) == 0:
                continue
            filt = ser[(ser.index >= eval_start) & (ser.index <= eval_end)]
            if len(filt) == 0:
                continue
            reindexed = filt.reindex(common_index).ffill().bfill()
            if reindexed.isna().all():
                continue
            ip = float(reindexed.iloc[0])
            if ip <= 0:
                continue
            bh_val = reindexed * (initial_capital / ip)
            bh_returns_pct_inst = ((bh_val - initial_capital) / initial_capital) * 100
            bh_series_by_inst[inst] = bh_returns_pct_inst
        bh_returns_pct = bh_series_by_inst[instruments_ordered[0]] if instruments_ordered and instruments_ordered[0] in bh_series_by_inst else None
    else:
        initial_price = eval_data.iloc[0]
        bh_shares = initial_capital / initial_price
        bh_values = eval_data * bh_shares
        bh_returns_pct = ((bh_values - initial_capital) / initial_capital) * 100
        bh_series_by_inst = {'Buy-and-Hold': bh_returns_pct}

    if bh_returns_pct is None:
        return None

    wallet_history = result.simulation.wallet_history
    portfolio_dates = [w.timestamp for w in wallet_history]
    interest_account = 0.0
    accrued = 0.0
    prev_date = None
    strategy_values = []
    for date, wallet_state in zip(portfolio_dates, wallet_history):
        if _is_new_month(date, prev_date):
            interest_account += accrued
            accrued = 0.0
        days = 1 if prev_date is None else (date - prev_date).days
        if days > 0 and (wallet_state.cash + interest_account) > 0:
            accrued += (wallet_state.cash + interest_account) * (daily_rate * days)
        strategy_values.append(wallet_state.total_value + interest_account)
        prev_date = date
    strategy_returns_pct = [((v - initial_capital) / initial_capital) * 100 for v in strategy_values]

    strategy_series = pd.Series(strategy_returns_pct, index=pd.DatetimeIndex(portfolio_dates))
    strategy_aligned = strategy_series.reindex(common_index, method='ffill')
    strategy_aligned = strategy_aligned.fillna(0.0)
    strategy_returns_pct_aligned = strategy_aligned.values

    alpha_over_time = []
    alpha_dates = []
    bh_arr = np.asarray(bh_returns_pct)
    for date, strategy_return in zip(portfolio_dates, strategy_returns_pct):
        try:
            date_for_lookup = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            bh_idx = common_index.get_indexer([date_for_lookup], method='nearest')[0]
            if 0 <= bh_idx < len(bh_arr):
                bh_return_at_date = bh_arr[bh_idx]
                alpha_over_time.append(strategy_return - bh_return_at_date)
                alpha_dates.append(date)
        except (IndexError, KeyError):
            continue
    if len(alpha_over_time) == 0:
        return None

    return AlphaOverTimeSeries(
        common_index=common_index,
        cash_only_returns_pct=cash_only_returns_pct,
        bh_returns_pct=bh_returns_pct,
        bh_series_by_inst=bh_series_by_inst,
        strategy_returns_pct_aligned=strategy_returns_pct_aligned,
        alpha_dates=alpha_dates,
        alpha_over_time=alpha_over_time,
        interest_rate_pa=interest_rate_pa,
    )

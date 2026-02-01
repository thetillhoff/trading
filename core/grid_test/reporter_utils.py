"""
Reporter utilities: constants, interest/date helpers, trades-to-DataFrame.

Used by reporter_analysis, reporter_base, and reporter_charts.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ..evaluation.walk_forward import WalkForwardResult

# Daily rate for 2% p.a. compound: (1.02)^(1/365.25) - 1 so that interest compounds to 2% per year
CASH_DAILY_RATE_2PA = (1.02 ** (1 / 365.25)) - 1

# When there are more instruments than this, legend shows only best-performing N + strategy (lines still plotted)
MAX_LEGEND_INSTRUMENTS = 10


def _daily_rate_from_pa(interest_rate_pa: float) -> float:
    """Convert annual interest rate (e.g. 0.02 for 2% p.a.) to daily compound rate. 0 -> 0."""
    if interest_rate_pa <= 0:
        return 0.0
    return (1.0 + interest_rate_pa) ** (1 / 365.25) - 1.0


def _is_new_month(current_date, prev_date):
    """True if current_date is in a different month/year than prev_date."""
    if prev_date is None:
        return False
    return (getattr(current_date, 'month', current_date.month), getattr(current_date, 'year', current_date.year)) != (
        getattr(prev_date, 'month', prev_date.month), getattr(prev_date, 'year', prev_date.year)
    )


def trades_to_dataframe(result: WalkForwardResult) -> pd.DataFrame:
    """
    Build a DataFrame of trades from a WalkForwardResult (same schema as save_trades_csv).

    Used by save_trades_csv and by the baseline snapshot test/generator.
    """
    if not result or not result.simulation.positions:
        return pd.DataFrame()
    rows = []
    for pos in result.simulation.positions:
        pnl_pct = (pos.pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0
        days_held = (pos.exit_timestamp - pos.entry_timestamp).days if pos.exit_timestamp and pos.entry_timestamp else None
        rows.append({
            'entry_date': pos.entry_timestamp.strftime('%Y-%m-%d') if pos.entry_timestamp else '',
            'entry_price': pos.entry_price,
            'exit_date': pos.exit_timestamp.strftime('%Y-%m-%d') if pos.exit_timestamp else '',
            'exit_price': pos.exit_price if pos.exit_price else '',
            'signal_type': pos.signal_type,
            'shares': pos.shares,
            'cost_basis': pos.cost_basis,
            'pnl': pos.pnl,
            'pnl_pct': pnl_pct,
            'status': pos.status.value if hasattr(pos.status, 'value') else str(pos.status),
            'target_price': pos.target_price if pos.target_price else '',
            'stop_loss': pos.stop_loss if pos.stop_loss else '',
            'days_held': float(days_held) if days_held is not None else np.nan,
            'rsi_value': pos.rsi_value if pos.rsi_value is not None else '',
            'ema_short': pos.ema_short if pos.ema_short is not None else '',
            'ema_long': pos.ema_long if pos.ema_long is not None else '',
            'macd_value': pos.macd_value if pos.macd_value is not None else '',
            'macd_signal': pos.macd_signal if pos.macd_signal is not None else '',
            'macd_histogram': pos.macd_histogram if pos.macd_histogram is not None else '',
            'indicator_confirmations': pos.indicator_confirmations,
            'original_signal_type': pos.original_signal_type,
            'certainty': pos.certainty,
            'risk_amount': pos.risk_amount,
            'risk_reward_ratio': pos.risk_reward_ratio,
            'projection_price': pos.projection_price if pos.projection_price else '',
            'position_size_method': pos.position_size_method,
            'quality_factor': getattr(pos, 'quality_factor', None),
            'trend_filter_active': pos.trend_filter_active,
            'trend_direction': pos.trend_direction,
            'instrument': pos.instrument if pos.instrument else '',
        })
    return pd.DataFrame(rows)

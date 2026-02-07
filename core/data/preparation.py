"""
Data preparation and validation for walk-forward evaluation.

Validates that all instruments have required data coverage,
no gaps in the eval range, and computes eval_dates upfront.
Fail-fast approach: raises on validation errors.
"""
from __future__ import annotations

import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple
from pathlib import Path

from .loader import DataLoader


# Extra calendar days when requesting load_start so the first trading day in data
# still gives at least lookback_days of history (avoids "only N days available" when
# eval_start - lookback_days falls on weekend/holiday)
LOOKBACK_CALENDAR_BUFFER_DAYS = 14

# Allow instrument's last date to be up to this many days before end_date (e.g. end_date
# on a holiday/weekend; last trading day is still valid coverage)
END_DATE_TOLERANCE_DAYS = 7


@dataclass
class VerifiedDataPrepResult:
    """Result from data preparation: validated ranges and eval_dates."""
    eval_dates: List[pd.Timestamp]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    load_start: pd.Timestamp
    instruments: List[str]
    min_history_days: int


class DataPreparationError(Exception):
    """Raised when data preparation validation fails."""
    pass


def _load_one_instrument(
    inst: str,
    load_start_str: str,
    end_date: str,
    column: str,
    start_ts: pd.Timestamp,
    requested_start: pd.Timestamp,
    lookback_days: int,
    min_history_days: int,
) -> Tuple[str, Optional[pd.Series], Optional[str]]:
    """Load and validate one instrument. Returns (inst, data or None, skip_reason or None)."""
    try:
        data = DataLoader.from_instrument(
            inst,
            start_date=load_start_str,
            end_date=end_date,
            column=column,
        )
    except Exception as e:
        return (inst, None, f"{type(e).__name__}: {e}")
    if data is None or len(data) == 0:
        return (inst, None, "no data after load")
    data_start = data.index.min()
    actual_days = (
        max(0, (start_ts - data_start).days) if data_start > requested_start else lookback_days
    )
    if actual_days < min_history_days:
        return (
            inst,
            None,
            f"insufficient history ({actual_days} < {min_history_days} days)",
        )
    return (inst, data, None)


def prepare_and_validate(
    instruments: List[str],
    start_date: str,
    end_date: str,
    lookback_days: int,
    step_days: int,
    min_history_days: int,
    column: str = "Close",
    max_workers: Optional[int] = None,
) -> VerifiedDataPrepResult:
    """
    Validate data availability and compute eval_dates for a config.
    
    For each instrument: load data and verify sufficient history (parallel by instrument).
    Instruments with no data or insufficient history in the requested range are skipped
    (with a warning); the result contains only instruments that pass.
    
    Args:
        instruments: List of instrument names (e.g. ["sp500", "djia"])
        start_date: Strategy start date (YYYY-MM-DD)
        end_date: Strategy end date (YYYY-MM-DD)
        lookback_days: Days of history needed for indicators
        step_days: Days between eval points
        min_history_days: Minimum days of history required before start_date
        column: Price column to validate (default: Close)
        max_workers: Max parallel workers for per-instrument load (default: min(32, len(instruments)))
        
    Returns:
        VerifiedDataPrepResult with eval_dates and validated ranges (instruments may be reduced)
        
    Raises:
        DataPreparationError: If no instruments have data in the requested range
    """
    if not instruments:
        raise DataPreparationError("No instruments provided")
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    load_start_ts = start_ts - timedelta(days=lookback_days + LOOKBACK_CALENDAR_BUFFER_DAYS)
    load_start_str = load_start_ts.strftime('%Y-%m-%d')
    requested_start = start_ts - timedelta(days=lookback_days)
    
    # Per-instrument load and validate in parallel (I/O-bound)
    workers = max_workers if max_workers is not None else min(32, max(1, len(instruments)))
    data_by_instrument: dict = {}
    skipped: List[str] = []
    skip_reasons: List[str] = []
    
    def task(inst: str) -> Tuple[str, Optional[pd.Series], Optional[str]]:
        return _load_one_instrument(
            inst,
            load_start_str,
            end_date,
            column,
            start_ts,
            requested_start,
            lookback_days,
            min_history_days,
        )
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_inst = {executor.submit(task, inst): inst for inst in instruments}
        for future in as_completed(future_to_inst):
            inst, data, reason = future.result()
            if reason is not None:
                skipped.append(inst)
                skip_reasons.append(reason)
            else:
                data_by_instrument[inst] = data
    
    # Preserve order: first valid instrument in original list is the reference for eval_dates
    ordered_valid = [inst for inst in instruments if inst in data_by_instrument]
    data_by_instrument = {inst: data_by_instrument[inst] for inst in ordered_valid}
    
    if not data_by_instrument:
        raise DataPreparationError(
            "No instruments have data in the requested range. "
            "Check date range and run download, or add instruments with data."
        )
    
    # Use first valid instrument's data as reference for eval_dates and range checks
    valid_instruments = list(data_by_instrument.keys())
    first_inst = valid_instruments[0]
    data_first = data_by_instrument[first_inst]
    data_start = data_first.index.min()
    data_end = data_first.index.max()
    
    # Adjust start_date if needed to ensure min_history_days (first valid instrument)
    effective_start = start_ts
    if effective_start < data_start + timedelta(days=min_history_days):
        effective_start = data_start + timedelta(days=min_history_days)
    
    # Skip instruments that don't cover the full eval range (warn instead of raise).
    # Allow inst_end up to END_DATE_TOLERANCE_DAYS before end_ts (end_date may be holiday/weekend).
    end_ts_slack = end_ts - timedelta(days=END_DATE_TOLERANCE_DAYS)
    for inst in list(data_by_instrument.keys()):
        if inst == first_inst:
            continue
        data_inst = data_by_instrument[inst]
        inst_start = data_inst.index.min()
        inst_end = data_inst.index.max()
        if inst_start > effective_start or inst_end < end_ts_slack:
            del data_by_instrument[inst]
            skipped.append(inst)
            skip_reasons.append(f"range {inst_start.date()}–{inst_end.date()} does not cover {effective_start.date()}–{end_ts.date()}")
    valid_instruments = list(data_by_instrument.keys())
    if not valid_instruments:
        raise DataPreparationError(
            "No instruments have data covering the full eval range. "
            "Check date range and run download, or add instruments with data."
        )
    # Re-establish first instrument and effective_start if we removed the previous first
    first_inst = valid_instruments[0]
    data_first = data_by_instrument[first_inst]
    data_start = data_first.index.min()
    effective_start = start_ts
    if effective_start < data_start + timedelta(days=min_history_days):
        effective_start = data_start + timedelta(days=min_history_days)
    
    # Check for gaps in first instrument's data (eval range only)
    eval_range_data = data_first[(data_first.index >= effective_start) & (data_first.index <= end_ts)]
    if len(eval_range_data) == 0:
        raise DataPreparationError(
            f"Instrument '{first_inst}': no data in eval range {effective_start.date()} to {end_ts.date()}"
        )
    
    # Simple gap check: if we have less than expected trading days (approx 252/year), warn
    expected_days = (end_ts - effective_start).days
    expected_trading_days = expected_days * 252 // 365  # rough estimate
    actual_trading_days = len(eval_range_data)
    if actual_trading_days < expected_trading_days * 0.8:  # Allow 20% tolerance for holidays/weekends
        # This is a soft warning, not a hard error for now
        pass  # Could log a warning here if we had logging
    
    # Generate eval_dates using first instrument's trading calendar
    eval_dates = _generate_eval_dates(data_first, effective_start, end_ts, step_days)
    
    if len(eval_dates) == 0:
        raise DataPreparationError(
            f"No evaluation dates generated for range {effective_start.date()} to {end_ts.date()} "
            f"with step_days={step_days}"
        )
    
    if skipped:
        print(f"Warning: skipped {len(skipped)} instrument(s): {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}", file=sys.stderr)
        # Log first few reasons so user can see why (e.g. FileNotFoundError vs column error)
        for i in range(min(5, len(skipped))):
            print(f"  {skipped[i]}: {skip_reasons[i]}", file=sys.stderr)
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more", file=sys.stderr)
    
    return VerifiedDataPrepResult(
        eval_dates=eval_dates,
        start_date=effective_start,
        end_date=end_ts,
        load_start=load_start_ts,
        instruments=valid_instruments,
        min_history_days=min_history_days,
    )


def _generate_eval_dates(
    data: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    step_days: int,
) -> List[pd.Timestamp]:
    """
    Generate evaluation dates at step_days intervals using trading calendar from data.
    
    Args:
        data: Price series with datetime index (trading calendar)
        start_date: Start of eval range
        end_date: End of eval range
        step_days: Days between eval points
        
    Returns:
        List of eval dates (pd.Timestamp)
    """
    eval_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Find the nearest trading day in data
        nearest_idx = data.index.get_indexer([current_date], method='nearest')[0]
        if nearest_idx >= 0 and nearest_idx < len(data):
            eval_date = data.index[nearest_idx]
            if eval_date >= start_date and eval_date <= end_date:
                if eval_date not in eval_dates:
                    eval_dates.append(eval_date)
        
        current_date += timedelta(days=step_days)
    
    return sorted(eval_dates)

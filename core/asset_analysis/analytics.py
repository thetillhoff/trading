"""
Returns, volatility, and correlation analytics from OHLCV data.

Takes Dict[instrument_name, DataFrame] and optional date range; uses Close for returns.
"""
import pandas as pd
from typing import Dict, Optional, Union
from datetime import datetime


def compute_returns(
    data_by_instrument: Dict[str, pd.DataFrame],
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Compute daily returns per instrument over the given window.

    Args:
        data_by_instrument: Map instrument name -> OHLCV DataFrame (datetime index).
        start_date: Start of window (inclusive). If None, use min index.
        end_date: End of window (inclusive). If None, use max index.
        column: Price column for returns (default Close).

    Returns:
        DataFrame with datetime index and one column per instrument (daily return).
    """
    if not data_by_instrument:
        return pd.DataFrame()

    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None

    series_list = []
    for name, df in data_by_instrument.items():
        if df.empty or column not in df.columns:
            continue
        s = df[column].sort_index()
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        ret = s.pct_change().dropna()
        ret.name = name
        series_list.append(ret)

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)


def compute_volatility_summary(
    data_by_instrument: Dict[str, pd.DataFrame],
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    column: str = "Close",
    window_days: int = 20,
) -> pd.DataFrame:
    """
    Per-instrument: daily returns, rolling volatility (std over window_days), and summary
    (annualized vol, avg daily return over the window).

    Args:
        data_by_instrument: Map instrument name -> OHLCV DataFrame.
        start_date: Start of window (inclusive).
        end_date: End of window (inclusive).
        column: Price column (default Close).
        window_days: Rolling volatility window (default 20).

    Returns:
        DataFrame with one row per instrument: annualized_vol, avg_daily_return, last_rolling_vol.
    """
    returns_df = compute_returns(
        data_by_instrument, start_date=start_date, end_date=end_date, column=column
    )
    if returns_df.empty:
        return pd.DataFrame(columns=["annualized_vol", "avg_daily_return", "rolling_vol"])

    rows = []
    for name in returns_df.columns:
        r = returns_df[name].dropna()
        if r.empty:
            rows.append({"instrument": name, "annualized_vol": None, "avg_daily_return": None, "rolling_vol": None})
            continue
        # Annualized vol: std of daily returns * sqrt(252)
        ann_vol = r.std() * (252 ** 0.5) if not pd.isna(r.std()) and r.std() != 0 else None
        avg_ret = r.mean()
        rolling = r.rolling(window_days, min_periods=min(window_days, len(r))).std()
        last_roll = rolling.iloc[-1] * (252 ** 0.5) if not rolling.empty and not pd.isna(rolling.iloc[-1]) else None
        rows.append({
            "instrument": name,
            "annualized_vol": float(ann_vol) if ann_vol is not None else None,
            "avg_daily_return": float(avg_ret) if avg_ret == avg_ret else None,
            "rolling_vol": float(last_roll) if last_roll is not None else None,
        })
    return pd.DataFrame(rows)


def compute_correlation_matrix(
    data_by_instrument: Dict[str, pd.DataFrame],
    start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Pairwise correlation of daily returns over the chosen window.

    Args:
        data_by_instrument: Map instrument name -> OHLCV DataFrame.
        start_date: Start of window (inclusive).
        end_date: End of window (inclusive).
        column: Price column (default Close).

    Returns:
        DataFrame correlation matrix (instruments x instruments).
    """
    returns_df = compute_returns(
        data_by_instrument, start_date=start_date, end_date=end_date, column=column
    )
    if returns_df.empty or returns_df.shape[1] < 2:
        return pd.DataFrame()
    return returns_df.corr()

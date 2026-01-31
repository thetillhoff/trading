"""Tests for configurable % p.a. interest logic in reporter: daily accrual, month-end compound."""
import pytest
import pandas as pd

from core.grid_test.reporter import (
    CASH_DAILY_RATE_2PA,
    _daily_rate_from_pa,
    _is_new_month,
)


def test_cash_daily_rate_compounds_to_2pa():
    """(1 + daily_rate)^365.25 should be ~1.02."""
    one_year = (1 + CASH_DAILY_RATE_2PA) ** 365.25
    assert 1.0195 <= one_year <= 1.0205


def test_is_new_month():
    """Month boundary detection."""
    jan31 = pd.Timestamp("2020-01-31")
    feb1 = pd.Timestamp("2020-02-01")
    assert _is_new_month(feb1, jan31) is True
    assert _is_new_month(jan31, jan31) is False
    assert _is_new_month(pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-10")) is False
    assert _is_new_month(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-01-31")) is True
    assert _is_new_month(pd.Timestamp("2021-01-01"), pd.Timestamp("2020-12-31")) is True


def test_cash_only_compound_one_year():
    """Cash-only with daily accrual and month-end payout yields ~2% over one year (default rate)."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    initial = 100.0
    balance = initial
    accrued = 0.0
    prev_date = None
    daily_rate = _daily_rate_from_pa(0.02)
    for current_date in dates:
        if _is_new_month(current_date, prev_date):
            balance += accrued
            accrued = 0.0
        days = 1 if prev_date is None else (current_date - prev_date).days
        if days > 0:
            accrued += balance * (daily_rate * days)
        prev_date = current_date
    final_value = balance + accrued
    return_pct = ((final_value - initial) / initial) * 100
    assert 1.98 <= return_pct <= 2.02


def test_configurable_rate_3pa_yields_approx_3pct():
    """Configurable rate 3% p.a. yields ~3% over one year."""
    daily_rate = _daily_rate_from_pa(0.03)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    initial = 100.0
    balance = initial
    accrued = 0.0
    prev_date = None
    for current_date in dates:
        if _is_new_month(current_date, prev_date):
            balance += accrued
            accrued = 0.0
        days = 1 if prev_date is None else (current_date - prev_date).days
        if days > 0:
            accrued += balance * (daily_rate * days)
        prev_date = current_date
    final_value = balance + accrued
    return_pct = ((final_value - initial) / initial) * 100
    assert 2.95 <= return_pct <= 3.05


def test_zero_rate_yields_no_interest():
    """interest_rate_pa 0 yields 0% return."""
    assert _daily_rate_from_pa(0.0) == 0.0
    assert _daily_rate_from_pa(-0.01) == 0.0

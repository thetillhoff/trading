"""
Pluggable signal rules for technical indicator signal generation.

Rules produce buy/sell "reasons" from indicator rows; the detector merges reasons
and builds TradingSignals. New rules can be added without changing detector logic.
"""
from typing import List, Tuple, Any, Protocol
import pandas as pd


class SignalRule(Protocol):
    """Protocol for a rule that evaluates one row and returns buy/sell reason strings."""

    def evaluate(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        config: Any,
    ) -> Tuple[List[str], List[str]]:
        """
        Evaluate rule at this row.

        Args:
            row: Current indicator row
            prev_row: Previous row (for crossovers)
            config: SignalConfig or StrategyConfig

        Returns:
            (buy_reasons, sell_reasons); either list may be empty
        """
        ...


class RsiRule:
    """RSI-based buy (exit oversold) and sell (enter overbought) reasons."""

    def evaluate(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        config: Any,
    ) -> Tuple[List[str], List[str]]:
        buy_reasons: List[str] = []
        sell_reasons: List[str] = []
        if pd.isna(row.get("rsi")):
            return buy_reasons, sell_reasons
        if prev_row.get("rsi_oversold") and not row.get("rsi_oversold"):
            buy_reasons.append(f"RSI exit oversold ({row['rsi']:.0f})")
        if not prev_row.get("rsi_overbought") and row.get("rsi_overbought"):
            sell_reasons.append(f"RSI enter overbought ({row['rsi']:.0f})")
        return buy_reasons, sell_reasons


class EmaRule:
    """EMA crossover buy/sell reasons."""

    def evaluate(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        config: Any,
    ) -> Tuple[List[str], List[str]]:
        buy_reasons: List[str] = []
        sell_reasons: List[str] = []
        if row.get("ema_bullish_cross"):
            buy_reasons.append("EMA bullish cross")
        if row.get("ema_bearish_cross"):
            sell_reasons.append("EMA bearish cross")
        return buy_reasons, sell_reasons


class MacdRule:
    """MACD histogram zero-cross buy/sell reasons."""

    def evaluate(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        config: Any,
    ) -> Tuple[List[str], List[str]]:
        buy_reasons: List[str] = []
        sell_reasons: List[str] = []
        if pd.isna(row.get("macd_histogram")) or pd.isna(prev_row.get("macd_histogram")):
            return buy_reasons, sell_reasons
        prev_hist = prev_row["macd_histogram"]
        curr_hist = row["macd_histogram"]
        if prev_hist < 0 and curr_hist >= 0:
            buy_reasons.append("MACD cross above zero")
        if prev_hist > 0 and curr_hist <= 0:
            sell_reasons.append("MACD cross below zero")
        return buy_reasons, sell_reasons


def get_technical_rules(config: Any) -> List[SignalRule]:
    """
    Return the list of technical indicator rules enabled by config.

    Order: RSI, EMA, MACD (matches legacy detector behavior).
    """
    rules: List[SignalRule] = []
    if getattr(config, "use_rsi", False):
        rules.append(RsiRule())
    if getattr(config, "use_ema", False):
        rules.append(EmaRule())
    if getattr(config, "use_macd", False):
        rules.append(MacdRule())
    return rules


def apply_trend_filter(
    buy_reasons: List[str],
    sell_reasons: List[str],
    row: pd.Series,
    config: Any,
) -> Tuple[List[str], List[str]]:
    """
    If use_trend_filter is enabled, drop buy reasons in bearish trend and sell reasons in bullish trend.
    """
    if not getattr(config, "use_trend_filter", False):
        return buy_reasons, sell_reasons
    ema_short = row.get("ema_short")
    ema_long = row.get("ema_long")
    is_bullish_trend = (
        ema_short > ema_long
        if (ema_short is not None and ema_long is not None)
        else None
    )
    if is_bullish_trend is None:
        return buy_reasons, sell_reasons
    out_buy = [] if not is_bullish_trend else buy_reasons
    out_sell = [] if is_bullish_trend else sell_reasons
    return out_buy, out_sell

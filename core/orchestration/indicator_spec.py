"""
Indicator spec hashing for cache reuse.

Extracts (indicator_type, params) from a strategy config and produces a stable
hash string. Configs that share the same indicator params (e.g. same Elliott Wave
min_confidence/min_wave_size) get the same spec hash and can reuse cached indicator
outputs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple


def _spec_hash(params: Dict[str, Any]) -> str:
    """Stable short hash for a params dict (sorted keys, JSON)."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def technical_spec_from_config(config: Any) -> Dict[str, Any]:
    """Params for TechnicalIndicators (RSI, EMA, MACD) from config."""
    return {
        "rsi_period": getattr(config, "rsi_period", 7),
        "rsi_oversold": getattr(config, "rsi_oversold", 25),
        "rsi_overbought": getattr(config, "rsi_overbought", 75),
        "ema_short_period": getattr(config, "ema_short_period", 20),
        "ema_long_period": getattr(config, "ema_long_period", 50),
        "macd_fast": getattr(config, "macd_fast", 12),
        "macd_slow": getattr(config, "macd_slow", 26),
        "macd_signal": getattr(config, "macd_signal", 12),
        "atr_period": getattr(config, "atr_period", 14),
        "volatility_window": getattr(config, "volatility_window", 20),
    }


def elliott_wave_spec_from_config(config: Any) -> Dict[str, Any]:
    """Params for Elliott Wave (regular) from config."""
    return {
        "min_confidence": getattr(config, "min_confidence", 0.65),
        "min_wave_size": getattr(config, "min_wave_size", 0.03),
    }


def elliott_wave_inverted_spec_from_config(config: Any) -> Dict[str, Any]:
    """Params for Elliott Wave inverted from config."""
    return {
        "min_confidence_inverted": getattr(config, "min_confidence_inverted", 0.65),
        "min_wave_size_inverted": getattr(config, "min_wave_size_inverted", 0.02),
    }


def indicator_spec_hash(indicator_type: str, params: Dict[str, Any]) -> str:
    """Stable hash for (indicator_type, params)."""
    payload = {"type": indicator_type, **params}
    return _spec_hash(payload)


def indicator_specs_needed_for_config(config: Any) -> List[Tuple[str, str]]:
    """
    List of (indicator_type, spec_key) that this config needs.

    spec_key is used as the cache path fragment (instrument_spec_key in contract).
    Indicator types: "technical", "elliott_wave", "elliott_wave_inverted".
    Returns only specs for indicators that are enabled on the config.
    """
    specs: List[Tuple[str, str]] = []
    if (
        getattr(config, "use_rsi", False)
        or getattr(config, "use_ema", False)
        or getattr(config, "use_macd", False)
    ):
        params = technical_spec_from_config(config)
        h = indicator_spec_hash("technical", params)
        specs.append(("technical", f"technical_{h}"))
    if getattr(config, "use_elliott_wave", False):
        params = elliott_wave_spec_from_config(config)
        h = indicator_spec_hash("elliott_wave", params)
        specs.append(("elliott_wave", f"elliott_wave_{h}"))
    if getattr(config, "use_elliott_wave_inverted", False) or getattr(
        config, "use_elliott_wave_inverted_exit", False
    ):
        params = elliott_wave_inverted_spec_from_config(config)
        h = indicator_spec_hash("elliott_wave_inverted", params)
        specs.append(("elliott_wave_inverted", f"elliott_wave_inverted_{h}"))
    return specs

"""
On-disk contract for the task-based evaluation pipeline.

Defines directory layout and canonical filenames so that Data, Indicators,
Signals, Simulation, and Outputs tasks read/write consistently. The orchestrator
and grid-search wrapper hold only path references; all data transport is on disk.

Layout (root = a workspace dir, e.g. temp or results/grid_search_<ts>):

  <root>/
    data/                          # Data task output
      prep_manifest.pkl             # VerifiedDataPrepResult + metadata
      <instrument>.parquet          # Per-instrument price series (safe name)
    indicator_cache/                # Indicators task output
      <instrument>_<indicator_spec_hash>/
        data.pkl                    # Indicator output (DataFrame or Series)
    configs/                        # Config staging per config
      <config_id>/                  # Safe config dir name
        signals/
          <instrument>.pkl          # Signals list per instrument
        result.pkl                  # Simulation task: WalkForwardResult (or serialized)
    results/                        # Per-config output (also used as config output_dir)
      <config_relative_path>/
        trades.csv                  # Canonical: all trades
        indicators.csv              # Canonical: indicator log
        results.csv                 # Canonical: one-row summary (was backtest_results_*)
        alpha_over_time.png
        value_gain_per_instrument_over_time.png
        scatter_pnl_pct_vs_duration.png
        scatter_confidence_risk_vs_pnl.png
        gain_per_instrument.png
        trades_per_instrument.png
        indicator_best_worst.png
        performance_timings.png
        comparison.png
        config.txt                  # Optional: human-readable config summary

For single-eval, root can be a single config output dir (results/<config_path>)
with no data/ or indicator_cache/ if they live elsewhere (e.g. shared temp).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List

# --- Canonical output filenames (per-config) ---
TRADES_CSV = "trades.csv"
INDICATORS_CSV = "indicators.csv"
RESULTS_CSV = "results.csv"
CONFIG_TXT = "config.txt"
CONFIG_YAML = "config.yaml"  # Timestamped copy of the original config

# --- Data task ---
DATA_DIR = "data"
PREP_MANIFEST_FILE = "prep_manifest.pkl"

# --- Indicator cache ---
INDICATOR_CACHE_DIR = "indicator_cache"

# --- Config staging (signals + simulation result) ---
CONFIGS_DIR = "configs"
SIGNALS_SUBDIR = "signals"
RESULT_FILE = "result.pkl"

# --- Per-config output charts (stable names) ---
CHART_ALPHA_OVER_TIME = "alpha_over_time.png"
CHART_VALUE_GAIN_PER_INSTRUMENT = "value_gain_per_instrument_over_time.png"
CHART_SCATTER_PNL_DURATION = "scatter_pnl_pct_vs_duration.png"
CHART_SCATTER_CONFIDENCE_RISK = "scatter_confidence_risk_vs_pnl.png"
CHART_GAIN_PER_INSTRUMENT = "gain_per_instrument.png"
CHART_TRADES_PER_INSTRUMENT = "trades_per_instrument.png"
CHART_INDICATOR_BEST_WORST = "indicator_best_worst.png"
CHART_PERFORMANCE_TIMINGS = "performance_timings.png"
CHART_COMPARISON = "comparison.png"

CHART_FILENAMES: List[str] = [
    CHART_ALPHA_OVER_TIME,
    CHART_VALUE_GAIN_PER_INSTRUMENT,
    CHART_SCATTER_PNL_DURATION,
    CHART_SCATTER_CONFIDENCE_RISK,
    CHART_GAIN_PER_INSTRUMENT,
    CHART_TRADES_PER_INSTRUMENT,
    CHART_INDICATOR_BEST_WORST,
    CHART_PERFORMANCE_TIMINGS,
    CHART_COMPARISON,
]


def safe_instrument_filename(name: str) -> str:
    """Safe filesystem name for instrument (e.g. BRK-B, AAPL)."""
    return re.sub(r"[^\w\-.]", "_", str(name))


def safe_config_dirname(name: str) -> str:
    """Safe filesystem directory name for config (e.g. grid_mtfon_ema8_filter_cert0.7)."""
    return re.sub(r"[^\w\-.]", "_", str(name))


def data_dir(root: Path) -> Path:
    """Path to the data task output directory."""
    return root / DATA_DIR


def prep_manifest_path(root: Path) -> Path:
    """Path to the prep manifest file (Data task output)."""
    return data_dir(root) / PREP_MANIFEST_FILE


def instrument_data_path(root: Path, instrument: str) -> Path:
    """Path to per-instrument price data (Data task output)."""
    return data_dir(root) / f"{safe_instrument_filename(instrument)}.parquet"


def indicator_cache_dir(root: Path) -> Path:
    """Path to the indicator cache directory."""
    return root / INDICATOR_CACHE_DIR


def indicator_output_dir(root: Path, instrument: str, indicator_spec_hash: str) -> Path:
    """Path to one (instrument, indicator_spec) cache entry. Task writes data.pkl inside."""
    return indicator_cache_dir(root) / f"{safe_instrument_filename(instrument)}_{indicator_spec_hash}"


def indicator_output_path(root: Path, instrument: str, indicator_spec_hash: str) -> Path:
    """Path to the serialized indicator output (data.pkl) for (instrument, spec)."""
    return indicator_output_dir(root, instrument, indicator_spec_hash) / "data.pkl"


def config_staging_dir(root: Path, config_id: str) -> Path:
    """Path to staging dir for one config (signals + result)."""
    return root / CONFIGS_DIR / safe_config_dirname(config_id)


def config_signals_dir(root: Path, config_id: str) -> Path:
    """Path to signals subdir for one config."""
    return config_staging_dir(root, config_id) / SIGNALS_SUBDIR


def config_signals_path(root: Path, config_id: str, instrument: str) -> Path:
    """Path to signals file for (config, instrument)."""
    return config_signals_dir(root, config_id) / f"{safe_instrument_filename(instrument)}.pkl"


def config_result_path(root: Path, config_id: str) -> Path:
    """Path to simulation result for one config."""
    return config_staging_dir(root, config_id) / RESULT_FILE


def config_output_dir(results_root: Path, config_relative_path: Path) -> Path:
    """Path to per-config output dir (trades.csv, indicators.csv, results.csv, charts)."""
    return results_root / config_relative_path


def trades_csv_path(output_dir: Path) -> Path:
    """Canonical path for trades CSV."""
    return output_dir / TRADES_CSV


def indicators_csv_path(output_dir: Path) -> Path:
    """Canonical path for indicators CSV."""
    return output_dir / INDICATORS_CSV


def results_csv_path(output_dir: Path) -> Path:
    """Canonical path for one-row summary results CSV."""
    return output_dir / RESULTS_CSV


def chart_path(output_dir: Path, chart_filename: str) -> Path:
    """Path for a chart file in the per-config output dir."""
    return output_dir / chart_filename


def timestamped_filename(base_name: str) -> str:
    """
    Add timestamp suffix to filename before extension.
    
    Examples:
        "trades.csv" -> "trades_20260205_123045.csv"
        "alpha_over_time.png" -> "alpha_over_time_20260205_123045.png"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_name)
    stem = path.stem
    suffix = path.suffix
    return f"{stem}_{timestamp}{suffix}"

pluggable/extensible rules, signals, indicators, instruments, etc.


# Trading Analysis Project

A comprehensive Python project for analyzing market data using Elliott Wave Theory,
technical indicators, signal detection, and trade evaluation.

## Overview

This project provides a complete toolkit for:

- **Data Collection**: Downloading historical data for any instrument (DJIA, S&P 500, DAX, Gold, EUR/USD, MSCI World)
- **Unified Indicators**: RSI, EMA, MACD, and Elliott Wave (all treated as indicators)
- **Trading Signals**: Configurable signal generation from any combination of indicators
- **Walk-Forward Evaluation**: Day-by-day backtesting (never uses future data)
- **Grid Search**: Systematic parameter and strategy comparison
- **Visualization**: Charts and CSV reports for analysis

## Quick Start

See all available commands:

```bash
make help

# One-time: download baseline data (S&P 500), then run evaluation
make download-baseline
make evaluate

# Download top 100 instrument candidates from ~500 available
make asset-analysis ARGS='--all-assets --fetch-metadata --analyze --top 100'

```

## Architecture

The system is built around a unified `core/` module that provides all functionality:

### Core Modules (`core/`)

- **`core/data/`**: Data loading and downloading
  - Downloads and caches historical data for multiple instruments from Yahoo Finance
  - `djia` - Dow Jones Industrial Average, `sp500` - S&P 500, `dax` - DAX 40, `gold` - Gold Futures, `eurusd` - EUR/USD, `msci_world` - MSCI World ETF

- **`core/indicators/`**: Technical analysis indicators
  - Elliott Wave detection with impulse/corrective wave patterns
  - RSI, EMA, MACD indicators with optimized parameters
  - Unified indicator interface for signal generation

- **`core/signals/`**: Signal generation and target calculation
  - Detects buy/sell signals from Elliott Wave patterns (Wave 2, 4, 5, B) and technical indicators
  - Calculates wave-specific Fibonacci targets and risk/reward-based stop-loss levels
  - Configurable signal combinations and confidence-based sizing

- **`core/evaluation/`**: Portfolio simulation and backtesting
  - Walk-forward backtesting with realistic capital management
  - Compares strategies against buy-and-hold with proper alpha calculation
  - Handles position sizing, risk management, and performance metrics

- **`core/shared/`**: Common types and configuration
  - Centralized default parameters for all indicators
  - Shared data types and enums

- **`core/grid_test/`**: Comparison and visualization
  - Generates performance charts and analysis reports
  - Creates trade timeline, scatter plots, and alpha-over-time visualizations

### CLI Interface (`cli/`)

Unified command-line interface for all operations:

- `make download` - Download market data
- `make evaluate` - Run strategy evaluation (defaults to `configs/baseline.yaml`, auto-generates charts)
- `make grid-search` - Compare multiple strategies from `configs/` (auto-parallel, auto-charts). Use `--output-dir` to direct outputs (e.g. when driving multi-period runs).
- `make hypothesis-tests` - Multi-period hypothesis tests via `cli.hypothesis`: grid-search per period (category/period selectable), then runs CSV-based analysis on the results dir and writes `analysis_report.md` plus aggregated CSVs there. To run analysis only on an existing dir: `make grid-search ARGS='--analyze results/hypothesis_tests_YYYYMMDD/'`.

## Quick Commands

```bash
# List available instruments
make scraper-list

# Download/update all instrument data
make scraper

# Generate Elliott Wave visualization
make visualize ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --elliott-waves"

# Analyze trading signals
make trading-signals ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"

# Evaluate trade performance
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"

# Generate multiple charts at once
make multi-charts ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --combined"

# Optimize filter parameters
make optimize-filters

# Run walk-forward backtest with baseline settings
make backtest

# Compare all preset strategies
make backtest-compare
```

## Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development history and design decisions
- **[ROADMAP.md](ROADMAP.md)**: Planned improvements and future features
- **Code Documentation**: All functionality is documented in the unified `core/` modules
  - See docstrings and comments in `core/data/`, `core/indicators/`, `core/signals/`, `core/evaluation/`, `core/grid_test/`

## Roadmap

For potential enhancements and future improvements, see [ROADMAP.md](ROADMAP.md).

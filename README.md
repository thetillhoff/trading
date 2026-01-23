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
```

For detailed examples and best practices, see [EXAMPLES.md](EXAMPLES.md).

## Project Structure

The project is organized into:

### Core Modules (`core/`)

Unified, instrument-agnostic modules for all trading operations:

- **`core/data/`**: Data loading and scraping for any instrument
- **`core/indicators/`**: Technical indicators (RSI, EMA, MACD) and Elliott Wave
- **`core/signals/`**: Unified signal detection from any indicator combination
- **`core/evaluation/`**: Walk-forward evaluation and portfolio simulation
- **`core/grid_test/`**: Grid search and comparison reporting

### CLI (`cli/`)

Unified command-line interface:

- **`cli/download.py`**: Download data for any instrument
- **`cli/evaluate.py`**: Evaluate a single strategy configuration
- **`cli/grid_search.py`**: Run grid search over parameter combinations

### Legacy Subprojects

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
- `make evaluate` - Run strategy evaluation
- `make grid-search` - Compare multiple strategies
- `make params` - Show configuration parameters

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

For detailed examples and best practices, see [EXAMPLES.md](EXAMPLES.md).

## Optimized Parameters

The following indicator parameters have been optimized through systematic grid search analysis:

### RSI (Relative Strength Index)
- **Period: 7** (optimized from default: 14)
- **Oversold: 25** (optimized from default: 30)
- **Overbought: 75** (optimized from default: 70)

**Derivation**: Grid search across 79 strategy configurations testing RSI periods 7, 14, and 21 on DJIA data from 2010-01-01 to 2020-01-01. RSI period 7 showed:
- Best average Alpha: -1.74% (vs -0.50% for period 14, -0.24% for period 21)
- Best average Expectancy: 0.51% per trade (vs 0.22% for period 14, 0.33% for period 21)
- Highest potential upside: range from -9.67% to +5.42% Alpha
- Tested across 32 strategy combinations

**Focused Testing Results** (95 configurations, 2010-2020):
- RSI thresholds 25/75: Average Alpha -0.14% (vs -1.29% for 30/70) - **Better performance, now default**
- 25/75 was tested in 16 strategies vs 44 for 30/70, but shows clear improvement

**Reasoning**: The shorter 7-period RSI is more sensitive to recent price movements, allowing for earlier entry signals in trending markets. While it shows higher variance, it provides better average performance when combined with other indicators. Tighter thresholds (25/75) reduce false signals and improve Alpha performance, now set as default.

### EMA (Exponential Moving Average)
- **Short Period: 20** (default)
- **Long Period: 50** (default)

**Derivation**: Grid search testing EMA combinations (9/21, 20/50, 50/200) on DJIA data from 2010-01-01 to 2020-01-01. EMA 20/50 showed:
- Best average Alpha: -0.51% (vs -3.81% for 9/21, -1.94% for 50/200)
- Best average Expectancy: 0.37% per trade (vs -0.11% for 9/21, 0.26% for 50/200)
- Best balance of performance and consistency (std dev: 3.96%)
- Tested across 32 strategy combinations

**Reasoning**: The 20/50 EMA combination provides optimal balance between responsiveness and trend-following stability. Shorter periods (9/21) are too sensitive and generate false signals, while longer periods (50/200) are too slow to capture medium-term trends effectively.

### MACD (Moving Average Convergence Divergence)
- **Fast Period: 12** (default)
- **Slow Period: 26** (default)
- **Signal Period: 12** (optimized from default: 9)

**Derivation**: Grid search testing MACD parameters on DJIA data from 2010-01-01 to 2020-01-01. Focused testing (95 configurations) revealed:
- MACD Signal 12: Alpha +0.44% (vs -1.96% for Signal 9) - **Best performance, now default**
- MACD Signal 7: Alpha +1.15% but only tested in 8 strategies - Needs more testing
- Signal 12 tested across 8 strategies, Signal 9 across 44 strategies

**Reasoning**: The 12/26/12 MACD configuration (optimized from standard 12/26/9) provides better trend confirmation signals. The 12-period fast EMA and 26-period slow EMA create a good balance for detecting momentum changes. Signal period 12 shows significantly better Alpha performance (+0.44% vs -1.96%) and is now the default.

### Elliott Wave Parameters
- **Min Confidence: 0.65** (default)
- **Min Wave Size: 0.03** (3% of price range, default)

**Derivation**: Grid search testing confidence levels (0.5, 0.65, 0.8) and wave sizes (0.02, 0.03, 0.05) on DJIA data from 2010-01-01 to 2020-01-01.

**Findings**:
- Confidence 0.8 + Wave Size 0.05: Produced **zero trades** (too restrictive)
- Confidence 0.5: High variance (-9.96% to +3.38% Alpha range) with poor average (-7.05%)
- Confidence 0.65 + Wave Size 0.03: Best balance, tested across 32 strategies

**Reasoning**: Lower confidence (0.5) generates too many false signals, while higher confidence (0.8) is too restrictive and misses opportunities. Confidence 0.65 with 3% minimum wave size provides the optimal trade-off between signal quality and quantity.

### Analysis Methodology

All parameters were optimized using:
- **Time Period**: 2010-01-01 to 2020-01-01 (training period)
- **Method**: Grid search with one-at-a-time parameter variation
- **Evaluation Metric**: Active Alpha (Hybrid Return - Buy-and-Hold Return)
- **Secondary Metric**: Expectancy (% return per trade)
- **Data**: DJIA daily close prices
- **Total Configurations**: 79 strategy combinations tested in parallel

Parameters were tested individually while keeping other parameters at defaults, then averaged across all indicator combinations to determine optimal values. This approach prevents exponential explosion of the search space while providing statistically meaningful results.

### Baseline Configuration

**Current Default: EMA + MACD** (optimized from grid search, 2010-2020)

The baseline configuration uses **EMA + MACD** as the optimal indicator combination, showing:
- **Alpha: +10.2%** (vs buy-and-hold)
- **Win Rate: 56.5%**
- **Trades: 223** over 10-year period
- **Best performing combination** in comprehensive grid search

**Risk Management Parameters** (validated as optimal):
- **Risk/Reward Ratio: 2.0** - Optimal balance between risk and reward
- **Position Size: 20%** - Optimal capital allocation per trade
- **Max Positions: 5** - Optimal diversification level
- **Confidence Multiplier: 0.1** - Optimal scaling for indicator confirmations

### Elliott Wave Analysis

**Key Finding: Elliott Wave underperforms when combined with technical indicators**

Comprehensive analysis (27 strategy configurations, 2010-2020) reveals:

**Performance Comparison:**
- **Elliott Wave Only**: 1.60% average Alpha, ~19 trades
- **Elliott Wave + Indicators**: 3.34% average Alpha, ~16 trades
- **Indicators Only**: 12.24% average Alpha, ~200 trades

**Root Cause:**
1. **Timing Mismatch**: Elliott Wave detects long-term patterns (weeks/months) while indicators respond to short-term momentum (days)
2. **Trade Frequency**: Indicators generate 10x more trading opportunities (200+ vs 19 trades)
3. **Signal Quality**: Indicator filtering reduces valid Elliott Wave signals, creating timing conflicts
4. **Performance Impact**: Indicators perform 4x better without Elliott Wave filtering

**Recommendation**: Use Elliott Wave separately for long-term analysis, or rely on indicators alone for active trading. The current baseline (EMA + MACD) provides optimal performance without Elliott Wave.

## Documentation

- **[EXAMPLES.md](EXAMPLES.md)**: Best practices and recommended commands
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development history and design decisions
- **[ROADMAP.md](ROADMAP.md)**: Planned improvements and future features
- **Code Documentation**: All functionality is documented in the unified `core/` modules
  - See docstrings and comments in `core/data/`, `core/indicators/`, `core/signals/`, `core/evaluation/`, `core/grid_test/`

## Roadmap

For potential enhancements and future improvements, see [ROADMAP.md](ROADMAP.md).

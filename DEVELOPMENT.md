# Development Guidelines

<!--
Common AI prompt:
Please take the next item from the ROADMAP.md and work to verify the hypothesis it represents
-->

## Project Overview

Python trading strategy backtesting system with Elliott Wave pattern detection, technical indicators (RSI, EMA, MACD), and walk-forward evaluation. Containerized with Docker for reproducibility.

**Note**: For general agent guidelines, Makefile usage, and code quality rules, see `AGENTS.md`. This file focuses on architecture and implementation details.

## Key Design Decisions

### Write & run tests

Write tests for all code. 80% coverage is the goal.
Run those tests to validate the code.

### Docker-First Approach

Use Docker containers for consistent environments, isolated dependencies, and reproducible builds.

### Data Caching Strategy

Save downloaded data to CSV files to reduce API calls, enable faster subsequent runs, and allow offline operation after initial download.

### Makefile for Common Operations

Use Makefiles for simplified workflows, consistent interface, and self-documenting commands via `make help`.

### Independent App Architecture

Each app has its own Dockerfile and requirements.txt for independence, scheduling flexibility, dependency isolation, and maintainability.

### Modular Architecture

Separate components for data loading, indicator calculation, signal detection, evaluation, and reporting for single responsibility, extensibility, testability, and reusability.

### Imports

- **core/** uses relative imports (e.g. `from ..shared.defaults import`). No `sys.path` manipulation.
- **cli/** and **scripts/** use absolute imports (`from core.xxx import`). Docker sets `PYTHONPATH=/app` so the project root is on the path.

### Standardized CLI Arguments

All scripts use consistent argument names, formats, defaults, and help text:

- `--start-date`, `--end-date`: Date ranges (YYYY-MM-DD)
- `--column`: Price column (default: Close)
- `--config`: YAML config file path
- `--config-dir`: Directory of YAML configs for grid search

## Architecture

### Data Flow

```
DataLoader (core/data/) → CSV cache from yfinance
    ↓
Indicators (core/indicators/) → RSI, EMA, MACD, ADX, Elliott Wave
    ↓
Signal Detection (core/signals/) → TradingSignal + targets/stops
    ↓
Evaluation (core/evaluation/) → WalkForwardEvaluator + PortfolioSimulator
    ↓
Reporting (core/grid_test/) → Charts + CSV results
```

### Module Responsibilities

- **core/data/**: Download and load OHLCV data (yfinance → CSV cache)
- **core/indicators/**: Calculate indicator values (no signals, just values). TechnicalIndicators includes ATR and volatility_20 (rolling return std) for risk/sizing/confirmation filter.
- **core/signals/**: Generate trading signals from indicator values
- **core/evaluation/**: Backtest strategies with realistic capital management. Trading costs configurable via `costs.trade_fee_pct` and `costs.trade_fee_absolute` (applied per side; result includes `total_trading_costs`). Optional `costs.trade_fee_min` and `costs.trade_fee_max` (absolute) clamp the fee per side to a minimum and maximum. Cash interest rate is configurable via `costs.interest_rate_pa` (default 0.02); reporter uses daily accrual and month-end payout so interest compounds; non-invested cash in strategy charts earns interest on its interest. **Non-invested cash**: The portfolio simulator does not apply interest; the reporter adds interest on cash for display in alpha_over_time and related charts. Non-invested cash earns only the configured interest rate (e.g. 2% p.a.), not instrument returns; displayed value uses balance/interest_account only (accrued not in account until month-end, so charts show monthly steps). **Market exposure**: The alpha-over-time figure includes a middle panel “Market exposure” (percentage of portfolio invested). **Multi-instrument**: When `config.instruments` has more than one symbol, walk-forward runs signal detection per instrument (each on its own history), merges and time-sorts signals, and simulates one portfolio with per-instrument prices for exits and PnL; buy-and-hold comparison uses the first instrument only. **Position size constraints**: `max_positions` (risk config) caps total open positions; `max_positions_per_instrument` (optional, default None) caps open positions **per instrument**. **Minimal position size**: `min_position_size` (risk config, optional) is an absolute value; when set, the simulator skips opening a position if the computed position capital would be below this minimum (relevant for multi-instrument; single-instrument still respects global `max_positions` only). **Position sizing**: `position_size_pct` (risk config) is the **maximum** fraction of portfolio per trade. Actual size = position_size_pct × quality_factor, with quality_factor ∈ [0, 1] from confirmations (and optionally confidence, volatility, risk/reward). Options: use_confidence_sizing (factor from confirmation score or count/3), use_confirmation_modulation (factor from confirmation_size_factors, normalized to 0–1), use_flexible_sizing, use_volatility_sizing. **Initial capital**: `initial_capital` (evaluation config, default 10000) sets starting portfolio capital for the backtest. **Feed history before timeframe**: When a timeframe (`start_date`/`end_date`) is specified in config, the evaluator requests data from `(start_date - lookback_days)` so that day-1 evaluation has full indicator history; if the cache does not have that far back, a warning is printed and evaluation continues with available data.
- **core/shared/**: Centralized defaults and shared types (single source of truth)
- **cli/**: Command-line interface (all operations via `make`)

### Extension Points

**Add New Indicator:**

1. Create class in `core/indicators/` implementing `Indicator` interface
2. Add parameters to `core/shared/defaults.py`
3. Update `TechnicalIndicators.calculate_all()` to include new indicator
4. Add CLI arguments in `cli/evaluate.py`

**Add New Strategy Preset:**

1. Add entry to `PRESET_CONFIGS` dict in `core/signals/config.py`
2. Specify which indicators to enable and their parameters
3. Accessible via `make evaluate ARGS="--preset your_preset"`

**Add New Performance Metric:**

1. Extend `SimulationResult` dataclass in `core/evaluation/portfolio.py`
2. Calculate metric in `PortfolioSimulator.simulate()`
3. Add to reporting in `core/grid_test/reporter.py`

### Baseline trades snapshot test

The test `tests/test_evaluation/test_baseline_trades_snapshot.py` checks that baseline evaluation on a short window (2012) produces the same trades as a stored snapshot. It requires djia data for 2012 and the golden file `tests/snapshots/baseline_trades_short.csv`. To create or refresh the snapshot after downloading data: `make baseline-snapshot-generate`. Create the snapshot directory first: `mkdir -p tests/snapshots`.

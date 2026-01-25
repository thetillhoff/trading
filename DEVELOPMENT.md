# Development Guidelines


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
- **core/indicators/**: Calculate indicator values (no signals, just values)
- **core/signals/**: Generate trading signals from indicator values
- **core/evaluation/**: Backtest strategies with realistic capital management
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

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

### Indicators vs Signals

**Indicators** are technical calculations on complete price history (RSI, EMA, Elliott Wave). They depend only on instrument + params, so they're **cacheable across runs** (same RSI(7) for SP500 is reused by all strategies). But they require **sequential computation** (EMA at day N depends on day N-1), so they cannot be split into chunks.

**Signals** are trading decisions evaluated at each date using a rolling window. They depend on indicators + strategy config (certainty, weights, risk/reward), so they're **strategy-specific and not cacheable**. But each eval_date is **independent**, so signal generation can be **split by year** and parallelized (5 yearly tasks instead of 1 monolithic task).

### Module Responsibilities

- **core/data/**: Download and load OHLCV data (yfinance → CSV cache). **Data preparation** (`preparation.py`): validates data availability and computes eval_dates upfront (fail-fast); used by evaluate and grid-search before signal generation.
- **core/indicators/**: Calculate indicator values (no signals, just values). TechnicalIndicators includes ATR and volatility_20 (rolling return std) for risk/sizing/confirmation filter.
- **core/signals/**: Generate trading signals from indicator values
- **core/evaluation/**: Backtest strategies with realistic capital management. Trading costs configurable via `costs.trade_fee_pct` and `costs.trade_fee_absolute` (applied per side; result includes `total_trading_costs`). Optional `costs.trade_fee_min` and `costs.trade_fee_max` (absolute) clamp the fee per side to a minimum and maximum. Cash interest rate is configurable via `costs.interest_rate_pa` (default 0.02); reporter uses daily accrual and month-end payout so interest compounds; non-invested cash in strategy charts earns interest on its interest. **Non-invested cash**: The portfolio simulator does not apply interest; the reporter adds interest on cash for display in alpha_over_time and related charts. Non-invested cash earns only the configured interest rate (e.g. 2% p.a.), not instrument returns; displayed value uses balance/interest_account only (accrued not in account until month-end, so charts show monthly steps). **Market exposure**: The alpha-over-time figure includes a middle panel “Market exposure” (percentage of portfolio invested). **Multi-instrument & parallelism**: When `config.instruments` has more than one symbol, walk-forward uses **small-jobs parallelism**: (1) data prep validates all instruments and computes eval_dates, (2) N parallel signal generation jobs (one per instrument, via `_signals_for_config_instrument_worker`), (3) merge signals, (4) one portfolio sim with per-instrument prices for exits and PnL; buy-and-hold comparison uses the first instrument only. **Grid-search parallelism**: M configs × N instruments = M×N parallel signal jobs; signals grouped by config, then one portfolio sim per config. Worker count is CPU-based (e.g. `cpu_count()-1`). Use `--workers` CLI arg to override. **Position size constraints**: `max_positions` (risk config) caps total open positions; `max_positions_per_instrument` (optional, default None) caps open positions **per instrument**. **Minimal position size**: `min_position_size` (risk config, optional) is an absolute value; when set, the simulator skips opening a position if the computed position capital would be below this minimum (relevant for multi-instrument; single-instrument still respects global `max_positions` only). **Position sizing**: `position_size_pct` (risk config) is the **maximum** fraction of portfolio per trade. Actual size = position_size_pct × quality_factor, with quality_factor ∈ [0, 1] from confirmations (and optionally confidence, volatility, risk/reward). Options: use_confidence_sizing (factor from confirmation score or count/3), use_confirmation_modulation (factor from confirmation_size_factors, normalized to 0–1), use_flexible_sizing, use_volatility_sizing. **Initial capital**: `initial_capital` (evaluation config, default 10000) sets starting portfolio capital for the backtest. **Feed history before timeframe**: When a timeframe (`start_date`/`end_date`) is specified in config, the evaluator requests data from `(start_date - lookback_days)` so that day-1 evaluation has full indicator history; if the cache does not have that far back, a warning is printed and evaluation continues with available data.
- **core/asset_analysis/**: Instrument metadata (yfinance), returns/volatility/correlation analytics, and candidate scoring (liquidity, volatility, correlation diversity). Metadata is cached in `data/instrument_metadata.json`; on rerun we load from file unless `--refresh-metadata` is passed. For discovered assets, use `--all-assets`: by default all sources (sp500, nasdaq100, dax, djia) are combined; use `--source sp500` (or nasdaq100, dax, djia) to limit to one source. Asset lists are cached per source in `data/available_assets_<source>.csv`, metadata in `data/available_assets_metadata.json`; `--refresh-assets` forces re-fetch of the list(s). With `--all-assets --analyze`, metadata and OHLCV are always for all discovered tickers (cache-first: metadata in `data/available_assets_metadata.json`, OHLCV per ticker in `data/tickers/` via `download_ticker`). Use `--top N` to write only the top N rows to the candidate_ranking CSVs (default: all). Without `--all-assets`, `--analyze` uses `download_instrument` and `DataLoader.from_instrument` (INSTRUMENTS only). Run `make asset-analysis` (e.g. `make asset-analysis ARGS='--all-assets --fetch-metadata --analyze --top 100'`). Outputs: `data/asset_analysis/volatility_summary.csv`, `correlation_matrix.csv`, `candidate_ranking.csv`, `candidate_ranking_filtered.csv`. The candidate list can be used to pick instruments for `config.instruments` or to extend the instrument set.
- **core/grid_test/**: Comparison and visualization for backtesting results. Reporter is split into: `reporter_utils.py` — constants (CASH*DAILY_RATE_2PA, MAX_LEGEND_INSTRUMENTS), helpers (\_daily_rate_from_pa, \_is_new_month), trades_to_dataframe; `reporter_analysis.py` — AlphaOverTimeSeries dataclass, compute_alpha_over_time_series; `reporter_charts.py` — ReporterChartsMixin with all generate*_ and *plot*_ chart methods; `reporter_base.py` — ComparisonReporter(ReporterChartsMixin) with **init**, paths, print*\*, save*\*, generate_analysis_report; `reporter.py` — re-exports for backward compatibility (import from here: ComparisonReporter, trades_to_dataframe, compute_alpha_over_time_series, etc.). Also: `analysis.py` (load results CSVs, analyze_results_dir), `grid_search.py` (generate_grid_configs).
- **core/shared/**: Centralized defaults and shared types (single source of truth)
- **cli/**: Command-line interface (all operations via `make`)

### Extension Points

**Add New Indicator:**

1. Create class in `core/indicators/` implementing `Indicator` interface (see `core/indicators/base.py`)
2. Add parameters to `core/shared/defaults.py`
3. Update `TechnicalIndicators.calculate_all()` to include new indicator
4. Add CLI arguments in `cli/evaluate.py` and wire into `SignalConfig` / config loader if used for signals

**Add New Signal Source (e.g. another detector):**

1. Implement detection in `core/signals/` or `core/indicators/` (e.g. Elliott Wave lives in indicators, detector orchestrates)
2. In `SignalDetector`: add a `_get_*_signals`-style method and call it from `detect_signals` / `detect_signals_with_indicators` when the config enables it. Reuse `_filter_signals_by_quality`, `_deduplicate_signals`, and target calculation.
3. Add config flags and params to `StrategyConfig` and `SignalConfig`; extend `_signal_config_from_strategy()` in `core/evaluation/walk_forward.py` so new strategies get the new options without duplicating config construction.

**Add New Technical Indicator Rule (RSI/EMA/MACD-style):**

1. In `core/signals/rules.py`: implement a class with `evaluate(row, prev_row, config) -> (buy_reasons: List[str], sell_reasons: List[str])` (satisfies `SignalRule` protocol).
2. Register it in `get_technical_rules(config)` so it runs when the corresponding config flag is set (e.g. `use_xyz`).
3. Add config fields to `StrategyConfig` / `SignalConfig` and wire from config loader if needed.

### Multi-Timeframe (MTF) Ensemble

**MTF Ensemble** allows using multiple timeframe periods simultaneously, where each contributes to signal confirmation with its own weight. This enables strategies like "primarily follow 4-week trend (weight 0.25) with secondary validation from 8-week trend (weight 0.05)".

**Configuration:**

MTF configuration is part of `indicator_weights.mtf` as a list of dicts:

```yaml
signals:
  use_multi_timeframe: true
  use_multi_timeframe_filter: false  # false = soft MTF (scoring only), true = hard filter
  indicator_weights:
    rsi: 0.6
    ema: 0.075
    macd: 0.075
    mtf:  # MTF ensemble
      - period: 4   # 4-week EMA
        weight: 0.25
      - period: 8   # 8-week EMA
        weight: 0.05
```

**Single period (non-ensemble):**

```yaml
mtf:
  - period: 4
    weight: 0.25
```

**Scoring & Filtering:**

- Each MTF config independently checks if weekly close ≥ weekly EMA (BUY) or ≤ (SELL)
- **Confirmation score**: Each period contributes `weight × (1 if confirmed else 0)` to the total score
- **Filter mode** (`use_multi_timeframe_filter: true`): Signal passes if weighted majority confirms (> 50% of total weight)
- Signal's `mtf_confirms` boolean is set based on weighted majority (used by filter and for analysis)

**Implementation:**

- `core/signals/config.py`: `indicator_weights` accepts `Union[float, List[Dict]]` for mtf value
- `core/signals/detector.py`: Computes multiple weekly EMAs, ensemble confirmations, and weighted scoring
- `core/indicators/technical.py`: `confirmation_weighted_score` takes `mtf_ensemble` parameter (list of {weight, confirmed})

**Breaking Change (Feb 2026):**

The `multi_timeframe_weekly_ema_period` field was removed. Existing configs must migrate:

Before:
```yaml
multi_timeframe_weekly_ema_period: 4
indicator_weights:
  mtf: 0.25
```

After:
```yaml
indicator_weights:
  mtf:
    - period: 4
      weight: 0.25
```

**Add New Strategy Preset:**

1. Add entry to `PRESET_CONFIGS` dict in `core/signals/config.py`
2. Specify which indicators to enable and their parameters
3. Accessible via `make evaluate ARGS="--preset your_preset"`

**Add New Performance Metric:**

1. Extend `SimulationResult` in `core/evaluation/portfolio_types.py`
2. Calculate metric in `PortfolioSimulator.simulate()` in `core/evaluation/portfolio.py`
3. Add to reporting in `core/grid_test/reporter_base.py` or `reporter_charts.py` as appropriate

**Extensibility and type modules (hypotheses / new strategies without rewriting):**

- **Types are split out** so new evaluators or strategies can reuse them: `core/evaluation/portfolio_types.py` (Position, SimulationResult, etc.), `core/evaluation/walk_forward_types.py` (WalkForwardResult, EvaluationSummary, etc.), `core/indicators/elliott_types.py` (Wave, WaveType, WaveLabel).
- **Single place for detector/simulator config:** `_signal_config_from_strategy(config)` and `_portfolio_simulator_from_config(config)` in `walk_forward.py` build `SignalConfig` and `PortfolioSimulator` from `StrategyConfig`; single- and multi-instrument eval both use these, so new config options only need to be added once.
- **Hypotheses:** Add new YAML configs (or grid-generated configs), run `make grid-search` or `make hypothesis-tests`; no code change needed for new parameter hypotheses. For new signal sources or metrics, follow the extension points above.

### Baseline trades snapshot test

The test `tests/test_evaluation/test_baseline_trades_snapshot.py` checks that baseline evaluation on a short window (2012) produces the same trades as a stored snapshot. It requires data for the instrument in `configs/baseline.yaml` (e.g. sp500) and the golden file `tests/snapshots/baseline_trades_short.csv`. To create or refresh the snapshot: `make download-baseline` then `make baseline-snapshot-generate`. Create the snapshot directory first: `mkdir -p tests/snapshots`. If you change the baseline instrument, regenerate the snapshot so the test passes.

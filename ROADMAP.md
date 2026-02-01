# Trading Strategy Roadmap

## Time ranges / events to analyze

covid_crash: "2019-01-01 - 2021-01-01"
recent_bull: "2015-01-01 - 2020-01-01"
recovery_period: "2009-01-01 - 2014-01-01"
housing_crisis: "2007-01-01 - 2010-01-01"
dotcom_crash: "2000-01-01 - 2003-01-01"
bear_market_long: "2000-01-01 - 2010-01-01"
bull_market_long: "2010-01-01 - 2020-01-01"

Evidence for hypotheses and baseline: [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md).
Current baseline: [configs/baseline.yaml](configs/baseline.yaml).

---

## Next Steps (Priority Order)

### High Priority
- **Small-jobs parallelism (evaluate + grid-search):** Today evaluate runs one multi-instrument walk-forward (parallel signal detection only within each eval date); grid-search runs one process per config. Unify on a single primitive: one task = produce all signals for (config, instrument) over the full walk-forward (same eval_dates for every instrument); no portfolio sim in the job. Aggregate: merge signal lists by time, run one portfolio sim per config. Evaluate: one config, N instruments → N parallel jobs, then merge + one sim. Grid-search: M×N jobs (config×instrument), group signals by config, merge + sim once per config. Period is in the config; walk-forward eval dates stay sequential inside each job.

- **restructure signals configuration**: Use a single configuration for indicators. If an indicator should be used, it has to be configured; if it's missing, it's not used. A minimal configuration contains at least a weight for the indicator and the necessary parameters. Filtering is via weighted score or min_confirmations/min_certainty.
  - This might affect the structure of this repo. From my understanding, the elliot wave config is used to generate signals. These are then filtered by the signal/indicator configuration. And then, based on those results, and other criteria, the position size is calculated. Only then, the trade is executed. Please create a flow diagram of the current process, vs this one. Note down pros and cons if there are differences.

- **asset analysis pt 2:**
  - Backtest scoring: run per-instrument backtest (e.g. baseline single-instrument), rank by return/Sharpe; optional flag (e.g. --backtest-score).
  - improve candidate filtering: add description of asset/company. fill gaps or remove those candidates. add some trading data like OHLCV etc, also export to same csv.

- **incorporate more data into algorithm:** Currently, only daily close data is used. Add more data like open, high, low, volume, etc. that's daily, and maybe even intraday data.

- **implement IBKR API**: Implement the IBKR API to get real-time data and execute trades. Sandbox first.

- **implement Alpaca API**: Implement the Alpaca API to get real-time data and execute trades. Sandbox first.

### Medium Priority – Strategy and execution

- **Exchange Traded Commodities:** Implement Exchange Traded Commodities (ETCs) as instruments. Add a few big ones, but only the physically backed ones.

- **Time-based trade constraints:** Min-holding time per trade; time-based stops (exit after N days). Use this to filter out trades that are too short-lived.
- **Indicator usage beyond confirmation:** Extend weighting to risk/stop-loss or other metrics (confirmation weighting already in baseline).

- **Hypothesis: Signal quality on full period (2000–2020):** Same signal_quality / conf×cert configs with start 2000, end 2020; confirm optimal min_certainty / min_confirmations over longer span.
- **Hypothesis: Multi-instrument with min_certainty:** Compare different values of. For example min_certainty 0.5 / 0.66; does selectivity improve multi-instrument results?

### Medium Priority – Config and code quality

- **Try around with indicator vs filters:** Some indicators could be used as filters, or as confirmation. And some filters could be used as indicators.

- **Independent review with fresh AI agent:**
  - **Test Coverage (61% overall):**
    - **Low Coverage (<50%):**
      - `core/data/download.py` (27%) — data downloading logic needs more tests
      - `core/data/scraper.py` (45%) — instrument scraping needs tests
      - `core/grid_test/reporter_charts.py` (30%) — chart generation hard to test; consider integration tests
      - `core/asset_analysis/discovery.py` (31%) — API mocking needed for source fetching
    - **Missing Coverage:**
      - `core/signals/target_calculator.py` (63%) — Fibonacci target calculation edge cases
      - Multi-instrument portfolio constraints (mostly covered, but edge cases remain)

  - **Potential Architecture & Design Improvements:**
    - **Logging Framework:** Replace ~50+ print() calls with Python `logging` module (levels: DEBUG, INFO, WARNING, ERROR). This enables:
      - Verbosity control via config/CLI flags
      - Log file output for long-running grid searches
      - Structured logging for analysis
      - Progress tracking without cluttering stdout
  - **Strategy & Feature Suggestions:**
    - **Multi-Timeframe Analysis as additional indicator:** Detect patterns on daily, confirm on weekly (mentioned in roadmap; high value for reducing false signals)
    - **Volume Confirmation:** Add volume indicators (OBV, VWMA) as optional confirmations. Many false breakouts happen on low volume.
    - **Adaptive Parameters:** Consider regime-aware parameter adjustment (e.g., wider stops in high volatility, tighter in low volatility)
    - **Correlation-Based Position Sizing:** When trading multiple instruments, reduce position size if instruments are highly correlated (avoids concentration risk)
    - **Add correlation to other instruments and market volume factors as indicators:** Use correlation and market volume to identify additional trade opportunities and signal strength / confirmation.
    - **Signal Strength Histogram:** Add visualization of signal distribution by confirmation score to identify threshold sweet spots
    - **Partial Exits:** Support scaling out of positions (e.g., take 50% at target, leave 50% with trailing stop)

  - **Testing Improvements:**
    - Add integration tests for end-to-end workflows (config → signals → evaluation → report)
    - Mock external APIs (yfinance) for deterministic tests
    - Property-based testing for financial calculations (use `hypothesis` library)
    - Regression tests for chart generation (image comparison or data structure validation)
    - Performance regression tests (ensure grid searches don't slow down over time)
  
  - **Code Simplifications:**
    - `walk_forward.py` has some duplicate logic between single-instrument and multi-instrument paths; consider unified flow
    - Config loading has fallback chains (CLI → YAML → defaults); consolidate into single precedence function
    - Some indicator confirmation logic is duplicated across buy/sell paths; extract to shared function
  
  - **Security & Robustness:**
    - Add input validation for CLI arguments (e.g., date format, numeric ranges)
    - Handle network failures gracefully in yfinance downloads (retry logic with exponential backoff)
    - Add file size limits when reading CSVs to prevent OOM on corrupted files
  
  - **Git & Workflow:**
    - Consider adding:
      - Pre-commit hooks for linting (black, flake8, mypy)
  
  - **Next Steps from This Review:**
    1. Split reporter.py into smaller modules (maintainability)
    2. Add logging framework (replaces prints)
    3. Increase test coverage to 80% (focus: download.py, scraper.py, target_calculator.py)
    4. Add integration tests for full workflows
    5. Consider multi-timeframe analysis (strategy improvement)

- **update readme:** Update the README.md file to reflect the current state of the project.

- **Config splitting:** Break StrategyConfig into sub-configs (Execution, Indicator, Risk) in the same file.
- **Logging:** Replace prints with a proper logging framework.

### Medium Priority – Elliott Wave

- **Stricter pattern rules:** Wave 4 cannot overlap Wave 1; Wave 3 not shortest; validate internal structure.
- **Multi-timeframe:** Detect on daily, confirm on weekly.

### Medium Priority – Risk and metrics

- **Risk-adjusted metrics:** Sharpe/Sortino, max drawdown, rolling window.
- **Advanced position sizing (optional):** Kelly, min/max limits, risk-per-trade; additive confidence is optimal so far; test max_positions when constraint binds (e.g. 0.5–0.7).

### Lower Priority – Regime (only if revisited)

- **Regime:** Current ADX/trend_vol regime rejected. "Flexible regime" (e.g. vol/range/trend) or regime-specific strategies only if we revisit regime with a new design.

### Lower Priority – Volatility

- **Volatility sizing:** Test volatility-based position sizing (ATR/price) if desired; filter was rejected, sizing untested.

### Lower Priority – CoT (Commitment of Traders)

- **CoT indicators:** Implement CoT as optional confirmation and verify usefulness. **Context:** CFTC weekly report of long/short positions (commercial vs speculator); often used as “smart money” or contrarian signal. **Limitations:** CoT applies only to futures—requires instrument→CFTC contract mapping (e.g. gold, eurusd); data is weekly (align to daily via ffill); separate data source and cache (CFTC CSV/API); not applicable to equity indices (djia, sp500) unless mapped to a futures contract (e.g. E-mini). Lower priority until other strategy/config work is done.

---

## Elliott Wave Enhancements

- **Stricter pattern rules (reduce false signals):** Wave 4 cannot overlap Wave 1; Wave 3 must not be shortest; validate internal structure.
- **Multi-timeframe:** Detect on daily, confirm on weekly.

---

## Future Enhancements (Lower Priority)

### Performance Metrics & Reporting

- **Advanced Metrics:**
  - Sharpe ratio, Sortino ratio tracking
  - Maximum drawdown analysis (calculated but not optimized for)
  - Rolling window performance
  - Performance by market regime (basic tracking exists)
  - Win/loss streak analysis
  - Multiple benchmark comparisons (not just buy-and-hold, e.g., MSCI World)
  - % win rate
  - % of total wallet amount that had market exposure over time
    o- **Grid Search Analysis:**
  - Pareto frontier visualization (optimal risk/reward trade-offs)
  - Automated parameter sensitivity analysis
  - Config diff tool (compare two strategies side-by-side)
  - Database storage (SQLite) for better result querying

### Additional Filters & Risk Management

- **Volume confirmation:** Require volume spike for signal validity
  - Volume-based indicators (OBV, VWMA, volume oscillator)
- **Support/Resistance:** Better entry/exit timing near key levels
- **Multi-timeframe:** Weekly trend + daily signals (detect on daily, confirm on weekly)
- **Advanced Stop Loss Types:**
  - Trailing stops (follow price up/down)
  - Time-based stops (exit after N days regardless of price)
  - Volatility-adjusted stops (wider in volatile periods)

### Portfolio & Multi-Asset Support

- **Multi-Asset Testing:** Test portfolios of multiple instruments
- **Correlation Analysis:** Cross-asset correlation tracking
- **Portfolio Optimization:** Optimal asset allocation across strategies
- **Data Quality:**
  - Validate downloaded data for gaps/errors
  - Handle missing values intelligently
  - Incremental data updates (only download new data, not full re-download)
  - Data integrity checks (detect splits, dividends, bad ticks)

### Real-time & Live Trading

- **Paper Trading:** Simulation mode with live data
- **Live Signal Generation:** Real-time signal detection
- **Trading System Integration:** Connect to broker APIs (future)
- **Trade Execution Types:** Support for long & short trades, plus limit orders, market orders, etc.

### Advanced Features (Experimental)

- Complex Elliott Wave patterns (diagonals, triangles)
- Wave degree identification across timeframes
- Machine learning for signal filtering
- Cross-validation for parameter optimization
- Quick backtest mode (faster with reduced accuracy for rapid iteration)
- Grid search resume capability (save progress, skip completed configs)
- Function refactoring (break down 200-300 line functions for maintainability)

---

## Status & Metrics

Evidence for tested hypotheses (position sizing, risk/reward, EW params, trend filter, regime, multi-instrument, inverted EW, etc.) is in [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md). Open items are listed under Next Steps above.

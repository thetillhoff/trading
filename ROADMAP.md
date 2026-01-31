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

- **restructure signals configuration**: Use a single configuration for indicators. If an indicator should be used, it has to be configured; if it's missing, it's not used. A minimal configuration contains at least a weight for the indicator and the necessary parameters. Filtering is via weighted score or min_confirmations/min_certainty.

- **asset analysis:**
  - Create a separate project for asset analysis. It's meant to find the best assets/instruments to use for trading.
  - Retrieve list of available assets/instruments and their metadata.
  - Retrieve data of available assets/instruments and analyze them.
  - Try to find common patterns and relationships between them.
  - Analyze their metadata, like market cap, company size, company ratings, etc.
  The goal is to identify metrics what makes an asset/instrument a good candidate for trading.
  And to prepare to add many more assets/instruments to the trading strategy.

- **implement IBKR API**: Implement the IBKR API to get real-time data and execute trades. Sandbox first.

- **implement Alpaca API**: Implement the Alpaca API to get real-time data and execute trades. Sandbox first.

### Medium Priority – Strategy and execution

- **Exchange Traded Commodities:** Implement Exchange Traded Commodities (ETCs) as instruments. Add a few big ones, but only the physically backed ones.

- **Time-based trade constraints:** Min-holding time per trade; time-based stops (exit after N days).
- **Indicator usage beyond confirmation:** Extend weighting to risk/stop-loss or other metrics (confirmation weighting already in baseline).

- **Hypothesis: Signal quality on full period (2000–2020):** Same signal_quality / conf×cert configs with start 2000, end 2020; confirm optimal min_certainty / min_confirmations over longer span.
- **Hypothesis: Multi-instrument with min_certainty:** Compare baseline multi-instrument vs variants with min_certainty 0.5 / 0.66; does selectivity improve multi-instrument results?

### Medium Priority – Config and code quality

- **Independent review with fresh AI agent:**
  - Review the code and tests specifically, but also this whole repository. Among regular review things, ensure the code and tests are correct, make sense and follow best practices. Examples:
    - Find unused code, make targets, tests, etc.
    - Suggest code- & test- & all other noteworthy improvements.
    - Suggest simplifications.
    - Suggest performance improvements.
    - Suggest new features.
    - Suggest strategy improvements.
    - Suggest improvements to AGENTS.md and DEVELOPMENT.md and the other markdown files.
  - Add your findings in this ROADMAP.md file.
  - "Clean up"/improve docstrings right away.

- **Custom signal rules:** Configurable rules without code changes; rule engine; indicator weights/priorities (partially covered by signal quality above).
- **Config splitting:** Break StrategyConfig into sub-configs (Execution, Indicator, Risk) in the same file.
- **Config validation:** Validate at creation (e.g. EMA short < long, valid ranges); fail fast with clear errors.
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

### Performance optimization opportunities (monitoring done)

- **Optimization:** Data is already loaded once per instrument (no per-step disk reads). Indicator calculation runs per (eval_date, instrument) slice from scratch; optional caching by (instrument, end_date) could avoid duplicate work for identical slices. Elliott Wave and target calculation have no shared cache; vectorizing/numba on hot paths only if profiling justifies.
- **Parallelization:** Per-indicator: run RSI, EMA, MACD, ADX, ATR in parallel inside `TechnicalIndicators.calculate_all`. Per-instrument: run instrument loop inside each eval_date in parallel (merge signals afterward).

### Advanced Features (Experimental)

- Complex Elliott Wave patterns (diagonals, triangles)
- Wave degree identification across timeframes
- Machine learning for signal filtering
- Cross-validation for parameter optimization
- Quick backtest mode (faster with reduced accuracy for rapid iteration)
- Grid search resume capability (save progress, skip completed configs)
- Signal rule extraction (separate SignalRule classes + rule engine)
- Function refactoring (break down 200-300 line functions for maintainability)

---

## Status & Metrics

Evidence for tested hypotheses (position sizing, risk/reward, EW params, trend filter, regime, multi-instrument, inverted EW, etc.) is in [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md). Open items are listed under Next Steps above.

# Trading Strategy Roadmap

## Time ranges / events to analyze

covid_crash: "2019-01-01 - 2021-01-01"
recent_bull: "2015-01-01 - 2020-01-01"
recovery_period: "2009-01-01 - 2014-01-01"
housing_crisis: "2007-01-01 - 2010-01-01"
dotcom_crash: "2000-01-01 - 2003-01-01"
bear_market_long: "2000-01-01 - 2010-01-01"
bull_market_long: "2010-01-01 - 2020-01-01"

For comprehensive hypothesis testing results, strategy comparisons, and detailed performance analysis, see [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md).

---

## Next Steps (Priority Order)

Evidence for completed work is in [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md).

### High Priority

- **Pre-trade analysis**: Are there indications on the trades that mark them as more probable to succeed or fail, before the trade is executed?
- **Time-based trade constraints:** Min-holding time per trade, time-based stops (exit after N days).
- **Implement Elliott Wave Enhancements**: Implement each of the enhancements to the Elliott Wave indicator mentioned further below.

### Medium Priority - Strategy

- **Regime redesign or removal:** ADX regime (25–40) shows no benefit (HYPOTHESIS_TEST_RESULTS). Try different regime logic (e.g. volatility, range vs trend) or remove.
- **Inverted Elliott Wave:** Does not improve alpha (HYPOTHESIS_TEST_RESULTS). Optional: investigate root cause or remove feature.

### Regime Detection Enhancements

- **Flexible Regime Detection:** More regime types beyond bull/bear
  - High volatility / low volatility
  - Range-bound / trending
  - Custom regime definitions
- **Regime-Specific Strategies:** Different strategies per regime
- **Volatility-detection:** Add volatility-based indicator for risk assessment and position sizing and confirmation of a trade.

- **Restructure indicator usage:** Enable config files to pick any indicator combination for any assessment metric, like risk, confidence, position size, stop loss, ..., with (multiplicative) weighting of each indicator.

### Medium Priority - Risk Management

- **Custom Signal Rules:** Configurable signal rules without code changes
  - Rule engine for flexible signal generation
  - Min confirmations parameter (not just boolean require_all)
  - Indicator weights/priorities
  - Signal strength scoring (quantify signal quality beyond binary)
- **Config Splitting:** Break StrategyConfig into sub-configs within the same file
  - ExecutionConfig (walk-forward, lookback, step days, ...)
  - IndicatorConfig (RSI, EMA, MACD, Elliott parameters, ...)
  - RiskConfig (position sizing, stops, risk/reward, ...)
- **Advanced position sizing:** Kelly, min/max limits, risk-per-trade limits, rebalancing. Additive confidence sizing is optimal (HYPOTHESIS_TEST_RESULTS); open: test max_positions at 0.5–0.7 if constraint ever binds.
- **Risk-adjusted metrics:** Sharpe/Sortino, max drawdown, rolling window.
- **Logging:** Replace prints with proper logging framework

---

## Elliott Wave Enhancements

### 1. Wave-Specific Targets

- **Wave 3:** 1.618-2.618× Wave 1
- **Wave 5:** Equal to Wave 1 or 0.618× Wave 3
- **Wave C:** Equal to Wave A or 1.618× Wave A

### 2. Stricter Pattern Rules (reduce false signals)

- Wave 4 cannot overlap Wave 1
- Wave 3 must not be shortest
- Validate internal structure

### 3. Parameter Optimization

- **Multi-timeframe:** Detect on daily, confirm on weekly.

## Future Enhancements (Lower Priority)

### Configuration & Extensibility

- **Trading Cost Configuration:**
  - Add % trade fee option in config
  - Add absolute trade fee option in config
  - Commission/slippage modeling for realistic backtesting
- **Interest/Cash Management:**
  - Add option in config to configure interest %/pa non-invested money earns (calculate on daily basis)
  - Verify if 2%pa line has compound interest yet (also the non-invested money should earn interest on its interest)

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
- **Evaluation Graphs:**
  - Total gain% vs MSCI World & vs 2%pa
  - Gain% vs duration of each trade in scatter plot
  - Confidence/risk vs gain per trade in scatter plot
  - Histogram of total % gained per stock (overall trades with it)
  - Histogram of amount of trades per stock
  - Result of other indicators compared to good trades, and bad trades - correlate which go well with it
  - Compare total % result against 2%pa and MSCI World, plus separate alpha graph between world and strategy
- **Grid Search Analysis:**
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
- **Position Size Constraints:**
  - Limit amount of parallel trades per instrument (not global)

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

### Infrastructure

- **Config Validation:** Validate strategy configs at creation
  - EMA short < long, positive values, valid ranges
  - Fail fast with clear error messages
- **Standardize Imports:** Fix inconsistent import styles
  - Remove sys.path.insert() where possible
  - Use relative imports within core/, absolute from cli/

### Advanced Features (Experimental)

- Complex Elliott Wave patterns (diagonals, triangles)
- Wave degree identification across timeframes
- Machine learning for signal filtering
- Cross-validation for parameter optimization
- Quick backtest mode (faster with reduced accuracy for rapid iteration)
- Grid search resume capability (save progress, skip completed configs)
- Signal rule extraction (separate SignalRule classes + rule engine)
- Function refactoring (break down 200-300 line functions for maintainability)

## Status & Metrics

Evidence for tested hypotheses (position sizing, risk/reward, EW params, trend filter, regime, multi-instrument, inverted EW, etc.) is in [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md). Open items are listed under Next Steps above.

---

**Current baseline:** `configs/baseline.yaml` (position 0.35, risk_reward 2.5). Evidence for these choices is in [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md) (Position Sizing and Risk Management, Risk/Reward Ratio). Optional upgrade: EW confidence 0.60–0.70, wave_length 0.30 (see Elliott Wave Parameter Sweep).

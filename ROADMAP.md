# Trading Strategy Roadmap

## Market analysis

Time ranges / events to analyze:

- covid crash
- dotcom crash
- housing crisis
- recent bull
- recent 2yr
- recovery period
- long bear market
- long bull market

For comprehensive hypothesis testing results, strategy comparisons, and detailed performance analysis, see [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md).

---

## Next Steps (Priority Order)

### High Priority - Immediate Testing

1. **Position Sizing Upper Bound Testing**
   - Test position sizing 0.35-0.4 to find upper bound (0.3 is optimal so far)
   - Test max_positions variations with position 0.3 (currently 10, test 5, 15, 20)
   - Test flexible position sizing (doesn't have to be full amount if confidence is lacking)
   - Test per-instrument position limits (not just global max_positions)

2. **Multi-Instrument Validation**
   - Test optimized configs on S&P 500, DAX, Gold
   - Validate if optimized strategies work across different markets
   - Use multi-instrument configs for unified portfolio testing
   - Test correlation analysis across instruments

### Medium Priority - Strategy Optimization

4. **Inverted Elliott Wave Investigation** ⚠️
   - Full period test: Inverted EW reduces alpha vs baseline (+106% vs +132%)
   - Inverted EW alone fails catastrophically (-30% alpha)
   - Investigate root cause: signal quality, parameter mismatch, or fundamental flaw?
   - Test alternative approaches: different inversion methods, parameter optimization
   - Consider removing inverted EW feature if no improvement found
   - **Status:** Not recommended for production - baseline remains optimal

5. **Regime Detection Investigation**
   - Investigate why regime detection shows no benefit (identical results to baseline)
   - Review regime detection implementation for bugs
   - Re-evaluate regime detection logic and thresholds
   - Consider removing or redesigning the approach
   - Test regime detection with different market conditions
   - Add flexible regime types (high/low volatility, range-bound/trending)

### Medium Priority - Risk Management

6. **Time-Based Trade Constraints**
   - Add min-holding time per trade in config (prevent 1-day stop-outs)
   - Test: 3-5 day minimum hold period
   - Expected: Higher win rate, ride winners longer
   - Add time-based stops (exit after N days regardless of price)

7. **Advanced Position Sizing & Constraints**
   - Kelly Criterion position sizing (size based on win rate and expectancy)
   - Min/max position size limits
   - Risk-per-trade limits (max % capital at risk per position)
   - Portfolio rebalancing (dynamic sizing based on current exposure)
   - If complete "wallet" is invested, no further trade possible until another finishes

8. **Risk-Adjusted Performance Metrics**
   - Sharpe ratio, Sortino ratio tracking and optimization
   - Target max Sharpe ratio instead of just returns
   - Maximum drawdown analysis and optimization
   - Rolling window performance analysis

---

## Elliott Wave Enhancements

### 1. Wave-Specific Targets
- **Wave 3:** 1.618-2.618× Wave 1
- **Wave 5:** Equal to Wave 1 or 0.618× Wave 3
- **Wave C:** Equal to Wave A or 1.618× Wave A

### 3. Stricter Pattern Rules (reduce false signals)
- Wave 4 cannot overlap Wave 1
- Wave 3 must not be shortest
- Validate internal structure

### 3. Parameter Optimization
- **Multi-timeframe:** Detect on daily, confirm on weekly
- **Note:** Wave size 0.01 and confidence 0.7 already tested (see HYPOTHESIS_TEST_RESULTS.md)


## Future Enhancements (Lower Priority)

### Configuration & Extensibility
- **Strategy Templates Enhancement:** Extend YAML config support
  - Version tracking for configurations
  - Config inheritance/composition (build from base configs)
- **Trading Cost Configuration:**
  - Add % trade fee option in config
  - Add absolute trade fee option in config
  - Commission/slippage modeling for realistic backtesting
- **Interest/Cash Management:**
  - Add option in config to configure interest %/pa non-invested money earns (calculate on daily basis)
  - Verify if 2%pa line has compound interest yet
- **Custom Indicators:** Plugin system for user-defined indicators
  - Document `Indicator` interface for external implementations
  - Drop-in indicator classes without code changes
- **Custom Signal Rules:** Configurable signal rules without code changes
  - Rule engine for flexible signal generation
  - Min confirmations parameter (not just boolean require_all)
  - Indicator weights/priorities
  - Signal strength scoring (quantify signal quality beyond binary)
- **Config Splitting:** Break StrategyConfig into sub-configs
  - IndicatorConfig (RSI, EMA, MACD, Elliott parameters)
  - RiskConfig (position sizing, stops, risk/reward)
  - ExecutionConfig (walk-forward, lookback, step days)

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
- **Backtesting Reports:**
  - HTML/PDF report generation
  - Enhanced trade visualization
  - Multi-period performance comparison
  - Trade journal export with full metadata for external analysis
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
- **Logging:** Replace prints with proper logging framework
- **Config Validation:** Validate strategy configs at creation
  - EMA short < long, positive values, valid ranges
  - Fail fast with clear error messages
- **Standardize Imports:** Fix inconsistent import styles
  - Remove sys.path.insert() where possible
  - Use relative imports within core/, absolute from cli/
- **Parameter Ranges Documentation:** Add tested ranges to defaults.py
  - Document min/max values for each parameter
  - Include optimization results as comments

### Regime Detection Enhancements
- **Flexible Regime Detection:** More regime types beyond bull/bear
  - High volatility / low volatility
  - Range-bound / trending
  - Custom regime definitions
- **Regime-Specific Strategies:** Different strategies per regime

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

**Targets Achieved:** ✅ 130%+ alpha (optimized config), ✅ Position sizing optimization successful, ✅ Risk management optimization successful

**Next Targets:** 
- Test position sizing 0.35-0.4 to find upper bound (0.3 is optimal so far)
- Test max_positions variations with position 0.3 (currently 10, test 5, 15, 20)
- Test on multiple instruments (S&P 500, DAX, Gold)
- Investigate why inverted EW alone fails (-30% alpha) or remove feature

**Usage:**
```bash
# Current best strategy (using config file)
make evaluate ARGS="--config configs/baseline.yaml --charts"

# Multi-instrument testing
make evaluate ARGS="--config configs/multi_instrument/ew_all_indicators_multi.yaml --charts"
```

For detailed performance metrics and hypothesis test results, see [HYPOTHESIS_TEST_RESULTS.md](HYPOTHESIS_TEST_RESULTS.md).

---

**Last Updated:** January 25, 2026  
**Current Baseline:** `configs/baseline.yaml` (position 0.3, risk_reward 2.5, +132% alpha)

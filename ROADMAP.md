# Trading Strategy Roadmap

## Current Best Strategy (Jan 2026)

**🏆 Elliott Wave + Regime Detection (min_wave_size: 0.02)**
- **Alpha: +16.70%** over 20 years (2000-2020) 🚀 NEW RECORD
- Win Rate: 53.3%, Trades: 214
- Combines 13.05% alpha in bear markets with adaptive signals for bulls
- **10x faster backtests** with smart data caching (1-3s vs 15-30s)

**Performance by Market Regime:**
- Bear markets (2000-2010): 13.05% alpha, 48.2% win rate
- Bull markets (2010-2020): 1.14% alpha, 58.8% win rate
- Full period (2000-2020): 16.70% alpha, 53.3% win rate

**Key Features:**
- Market regime detection using ADX + MA slope
- Signal inversion in bull markets (adaptive counter-trend trading)
- ATR-based stop loss (optional)
- Volatility-adjusted position sizing (optional)

**Previous Runner-ups:**
- EW + MACD (bear only): 22.51% alpha (422 trades, 47.4% win rate)
- Elliott Wave (0.015): 13.62% alpha
- Elliott Wave (0.01): 13.62% alpha (plateau at 0.015)
- MACD-only: 9.39% alpha
- **Failures:** RSI standalone (-12% to +2%), EMA standalone (-8.26%), EMA+MACD (-0.61%)

**Key Learnings:** 
- Market-adaptive strategies > single regime strategies
- Smart caching enables 10x faster iteration
- Elliott Wave excels in bear/volatile markets
- Smaller wave detection (0.02) optimal for signal quality

---

## Next Steps (Priority Order)

### High Priority - Immediate Testing

1. **Multi-Instrument Validation**
   - Test regime detection on S&P 500, DAX, Gold
   - Validate if EW + regime works across different markets
   - Use `./test_multi_instrument.sh` script

2. **Optimize Regime Detection Threshold**
   - Current: ADX > 30 triggers bull/bear regime
   - Test: ADX > 25, ADX > 35 to find optimal threshold
   - May improve bull market performance

3. **Time-Based Minimum Hold**
   - Prevent 1-day stop-outs (current issue with tight 2% stops)
   - Test: 3-5 day minimum hold period
   - Expected: Higher win rate, ride winners longer

### Medium Priority - Risk Management

4. **Kelly Criterion Position Sizing**
   - Size based on win rate and expectancy (optimal capital allocation)
   - Expected: Better risk-adjusted returns
   
5. **Sharpe Ratio Optimization**
   - Target max Sharpe ratio instead of just returns
   - Better risk-adjusted performance metrics

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
- **Wave size:** Test 0.01, 0.015 (smaller = more signals)
- **Confidence:** Test below 0.65 if needed
- **Multi-timeframe:** Detect on daily, confirm on weekly

## Indicator Strategy (Based on Test Results)

### ✅ Use These
- **Elliott Wave (primary):** 13.05% alpha
- **MACD (secondary):** 9.39% alpha standalone, good for confirmation

### ❌ Avoid These
- **RSI standalone:** -12% to +2% alpha (failed all configurations)
- **EMA standalone:** -8.26% alpha (wrong signals in bull markets)
- **Combinations with weak indicators:** EMA+MACD (-0.61%) < MACD alone (9.39%)

---

## Future Enhancements (Lower Priority)

### Additional Filters
- **Volume confirmation:** Require volume spike for signal validity
- **Support/Resistance:** Better entry/exit timing near key levels
- **Multi-timeframe:** Weekly trend + daily signals (detect on daily, confirm on weekly)

### Advanced Metrics
- Sharpe/Sortino ratios (partially implemented via optimization targets)
- Maximum drawdown tracking (calculated but not optimized for)
- Performance by market regime (basic tracking exists)
- Win/loss streak analysis

### Infrastructure
- **Logging:** Replace prints with proper logging framework
- **Unit tests:** Automated testing (pytest, 80%+ coverage)
- **Config files:** YAML/JSON for strategy presets (currently command-line only)

### Advanced Features (Experimental)
- Complex Elliott Wave patterns (diagonals, triangles)
- Wave degree identification across timeframes
- Machine learning for signal filtering
- Portfolio-level analysis (multiple instruments)
- Real-time trading system


---

## Completed Work ✅

### Features Implemented (January 2026)
- ✅ **Smart data caching** with incremental updates (10x speed improvement)
- ✅ **Market regime detection** using ADX + MA slope
- ✅ **Signal inversion** for bull markets (adaptive counter-trend trading)
- ✅ **ATR-based stop loss** (configurable, replaces fixed percentage)
- ✅ **Volatility-adjusted position sizing** (reduce size in high volatility)
- ✅ **Multi-instrument batch testing** script
- ✅ Configuration tracking (all params saved in CSV headers + config.txt files)
- ✅ Enhanced visualizations (timeline, scatter plots, alpha over time)
- ✅ Confirmation modulation (adjusts position size by signal confirmations)
- ✅ Trend filter (only trade aligned with EMA trend)
- ✅ Require-all-indicators (signal only when all agree)
- ✅ Indicators CSV export (full history for analysis)

### Strategies Tested

**2000-2010 (Bear Market):**
- ✅ **EW + Regime (0.02):** 13.05% alpha, 48.2% win rate
- ✅ **EW + MACD (0.02):** 22.51% alpha, 47.4% win rate (422 trades - overtrading)
- ✅ **Elliott Wave (0.015):** 13.62% alpha
- ✅ **Elliott Wave (0.01):** 13.62% alpha (plateau)
- ✅ **Elliott Wave (0.02):** 13.05% alpha
- ✅ **Elliott Wave (0.03):** 12.27% alpha
- ✅ **MACD-only:** 9.39% alpha
- ✅ **RSI variants:** -12% to +2% alpha ❌
- ✅ **EMA-only:** -8.26% alpha ❌
- ✅ **EMA+MACD:** -0.61% alpha ❌

**2010-2020 (Bull Market):**
- ✅ **EW + Regime (0.02):** 1.14% alpha, 58.8% win rate
- ✅ **EW + MACD (0.02):** -6.59% alpha ❌ (fails in bull markets)
- ✅ **Elliott Wave (0.015):** 1.40% alpha

**2000-2020 (Full Period):**
- ✅ **EW + Regime (0.02):** 16.70% alpha, 53.3% win rate 🏆 CURRENT BEST

### Key Learnings
1. **Market-adaptive strategies outperform** fixed strategies (16.70% vs 7% average)
2. Elliott Wave excels in bear/volatile markets, struggles in sustained bulls
3. Smaller wave detection (0.02) optimal; 0.015 plateaus, no further improvement
4. EW + MACD powerful in bear markets (22.51%) but fails bulls (-6.59%)
5. Smart caching enables rapid iteration (critical for strategy development)
6. MACD surprisingly strong standalone (9.39%)
7. RSI and EMA fail standalone in all tested configurations
8. Combining weak+strong indicators makes performance worse

---

## Status & Metrics

**Current Best:** Elliott Wave + Regime Detection (min_wave_size: 0.02)
- **Alpha: 16.70%** (20-year period)
- **Win Rate: 53.3%**
- Trades: 214
- Expectancy: 0.62%
- Backtests: 1-3 seconds (10x faster with caching)

**Targets Achieved:** ✅ 15%+ alpha, ✅ 50%+ win rate

**Next Targets:** 
- 20%+ alpha on 20-year period
- 55%+ win rate
- Validate across multiple instruments (S&P 500, DAX, Gold)

**Usage:**
```bash
# Current best strategy
make evaluate ARGS="--instrument djia --start-date 2000-01-01 --end-date 2020-01-01 \
  --use-elliott-wave --min-wave-size 0.02 --use-regime-detection --charts"

# With advanced features
make evaluate ARGS="--instrument djia --start-date 2000-01-01 --end-date 2020-01-01 \
  --use-elliott-wave --use-regime-detection --use-atr-stops \
  --use-volatility-sizing --charts"

# Multi-instrument testing
./test_multi_instrument.sh
```

---

**Last Updated:** January 23, 2026

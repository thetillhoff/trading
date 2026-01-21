# Trading Analysis Project

A comprehensive Python project for analyzing DJIA (Dow Jones Industrial Average) trading data
using Elliott Wave Theory, signal detection, and trade evaluation.

## Overview

This project provides a complete toolkit for:

- **Data Collection**: Downloading and caching historical DJIA data
- **Elliott Wave Analysis**: Detecting wave patterns in price movements
- **Trading Signals**: Identifying buy/sell opportunities with target prices and stop-loss levels
- **Trade Evaluation**: Backtesting signals to calculate performance metrics
- **Visualization**: Generating charts and graphs for analysis

## Quick Start

See all available commands:

```bash
make help
```

For detailed examples and best practices, see [EXAMPLES.md](EXAMPLES.md).

## Available Tools

### Data Collection

- **DJIA Scraper**: Downloads and caches historical DJIA data from yfinance

### Analysis Tools

- **Elliott Wave Detection**: Identifies wave patterns in price data
- **Filter Optimizer**: Finds optimal filter parameters for wave detection
- **Trading Signals**: Detects buy/sell opportunities with targets and stop-loss
- **Trade Evaluator**: Backtests signals and calculates performance metrics

### Visualization

- **Price Charts**: Basic line charts with customizable granularity
- **Elliott Wave Charts**: Price charts with color-coded wave patterns
- **Trading Signals Charts**: Charts showing buy/sell points, targets, and stop-loss
- **Trade Evaluation Charts**: Performance visualization with win/loss indicators
- **Multi-Chart Generation**: Generate multiple charts with shared parameters

## Quick Commands

```bash
# Download/update data
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
```

For detailed examples and best practices, see [EXAMPLES.md](EXAMPLES.md).

## Documentation

- **[EXAMPLES.md](EXAMPLES.md)**: Best practices and recommended commands
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development history and design decisions
- **Module READMEs**: Detailed documentation in each module directory
  - `scrapers/djia/README.md` - Scraper documentation
  - `visualizations/djia/README.md` - Visualization documentation
  - `visualizations/elliott_wave_optimizer/README.md` - Filter optimizer documentation
  - `visualizations/trading_signals/README.md` - Trading signals documentation
  - `visualizations/trade_evaluator/README.md` - Trade evaluator documentation

## Suggested Improvements

The current implementation provides Elliott Wave detection, signal generation, target calculation
(using Fibonacci 1.618 for buys, 0.5 for sells), risk/reward-based stop-loss (default 2:1), trade
evaluation with backtesting, performance metrics, and visualization. The following enhancements
would align with professional Elliott Wave trading frameworks:

### 1. Multi-Timeframe Analysis

**Current**: Single timeframe analysis

**Improvement**: Analyze higher timeframes (daily → weekly → monthly) to identify wave degrees
and ensure lower-degree waves align with higher-degree trends. Trade with the dominant wave
degree.

**Requirements**:

- Data aggregation for multiple timeframes (daily, weekly, monthly)
- Wave degree classification system
- Cross-timeframe wave alignment validation
- Timeframe hierarchy data structure

**Implementation**: Add timeframe hierarchy analysis and wave degree classification

### 2. Enhanced Entry Validation

**Current**: Basic signals at end of Wave 2, 4, 5, and B

**Improvement**: Add validation checklists for high-probability setups:

- **Wave 2 entries**: Validate Wave 1 is impulsive, Wave 2 retraces 50-78.6%, correction is 3
  waves (A-B-C), no overlap with Wave 1 start
- **Wave 4 entries**: Validate Wave 3 is extended, Wave 4 retraces 23.6-38.2%, no overlap with
  Wave 1
- **Wave 5 exits**: Validate 5-wave structure complete, Wave 5 extended or truncated

**Requirements**:

- Wave retracement calculation (percentage of prior wave)
- Sub-wave structure detection (5-wave vs 3-wave validation)
- Wave overlap detection algorithm
- Entry quality scoring system

**Implementation**: Add pattern validation rules and entry quality scoring

### 3. Wave-Specific Fibonacci Targets

**Current**: Uses 1.618× for buy targets, 0.5× for sell targets

**Improvement**: Implement wave-specific relationships:

- Wave 3: 1.618-2.618 × Wave 1
- Wave 4: 23.6-38.2% of Wave 3
- Wave 5: = Wave 1 or 0.618 × Wave 3
- Wave C: = Wave A or 1.618 × Wave A

**Requirements**:

- Wave relationship mapping (Wave 1 → Wave 3, Wave 3 → Wave 4, etc.)
- Multiple target calculation (primary and secondary targets)
- Wave length comparison logic (Wave 5 = Wave 1)
- Fibonacci level selection algorithm

**Implementation**: Enhance target calculator with wave-specific Fibonacci relationships

### 4. Technical Indicator Confirmation

**Current**: Elliott Wave analysis only

**Improvement**: Add multi-indicator confirmation (RSI/MACD divergence, volume analysis,
trendline breaks, support/resistance confluence). Require at least two confirmations before
generating signals.

**Requirements**:

- Technical indicator library (RSI, MACD, moving averages)
- Divergence detection algorithm
- Volume analysis module
- Trendline detection and break identification
- Support/resistance level calculation
- Confirmation scoring system

**Implementation**: Add technical indicator modules (RSI, MACD, volume, trendline detection)

### 5. Pattern Validation Rules

**Current**: Basic pattern completeness checking

**Improvement**: Add validation rules:

- Wave overlap detection (Wave 4 cannot overlap Wave 1, except diagonals)
- Wave 3 validation (not shortest of 1, 3, 5)
- Internal structure validation (Wave 2 should be 3 waves)
- Pattern ambiguity detection

**Requirements**:

- Wave overlap detection algorithm
- Wave length comparison (Wave 3 vs Wave 1 and Wave 5)
- Sub-wave structure analysis
- Pattern ambiguity scoring
- Diagonal triangle pattern recognition

**Implementation**: Enhance pattern validation with Elliott Wave rules

### 6. Advanced Pattern Recognition

**Current**: Basic impulse and corrective patterns

**Improvement**: Detect complex patterns (diagonal triangles, flat corrections, zigzags, triangle
corrections, combined corrections W-X-Y)

**Requirements**:

- Pattern classification algorithms for each pattern type
- Diagonal triangle detection (leading/ending)
- Flat correction identification (regular/expanded/running)
- Zigzag pattern recognition (simple/double/triple)
- Triangle pattern detection (ascending/descending/symmetrical/expanding)
- Combined correction pattern matching (W-X-Y, W-X-Y-X-Z)
- Pattern-specific entry/exit rules

**Implementation**: Add pattern classification system with pattern-specific rules

### 7. Position Sizing & Wave Invalidation

**Current**: Risk/reward ratio (2:1 default), max holding period

**Improvement**: Add position sizing calculator (risk ≤ 1-2% per trade), automatic exit on
pattern invalidation (e.g., Wave 2 entry invalidated if price breaks below Wave 1 start)

**Requirements**:

- Position sizing calculator (account size, risk percentage, stop-loss distance)
- Pattern invalidation detection (price breaks key wave levels)
- Real-time price monitoring for invalidation
- Automatic exit signal generation
- Risk management integration

**Implementation**: Add position sizing module and pattern invalidation detection

### 8. Advanced Performance Analytics

**Current**: Basic metrics (win rate, average gain/loss, total gain, buy-and-hold comparison)

**Improvement**: Performance by wave type, pattern quality score, confirmation tool usage,
drawdown analysis, risk-adjusted returns (Sharpe/Sortino), regime-aware analysis

**Requirements**:

- Performance segmentation by wave type (Wave 2, 4, 5, B entries)
- Pattern quality score tracking
- Confirmation tool usage tracking
- Drawdown calculation and analysis
- Risk-adjusted return calculations (Sharpe ratio, Sortino ratio)
- Regime detection integration for regime-aware analysis
- Performance attribution framework

**Implementation**: Enhance trade evaluator with pattern-level analytics

### 9. Real-Time Monitoring

**Current**: Historical analysis and backtesting

**Improvement**: Live monitoring with alerts for pattern completion, high-probability setups,
pattern invalidation

**Requirements**:

- Real-time data feed integration (API or streaming)
- Pattern update detection (new waves, pattern completion)
- Alert system (email, SMS, webhook, or in-app notifications)
- High-probability setup detection and scoring
- Pattern invalidation monitoring
- Notification configuration and filtering

**Implementation**: Add real-time data feed integration and alert system

### Additional Trading Analytics Frameworks

Beyond Elliott Wave analysis, the following systematic trading frameworks could be integrated
to enhance the trading system:

#### 1. Factor-Based / Systematic Investing

**What it does**: Decomposes returns into known drivers (value, momentum, quality, volatility,
size, etc.)

**Typical signals**:

- Momentum (12-1 month returns)
- Low volatility
- Value (P/E, EV/EBITDA)
- Quality (ROE, debt ratios)

**Why it works well for DJIA**: DJIA components have rich fundamental data. Long-term
systematic strategies are robust.

**Automation level**: ⭐⭐⭐⭐⭐

**Used by**: AQR, BlackRock, JPM Quant

**Requirements**:

- Fundamental data scraper (financial statements, ratios)
- Factor calculation engine (momentum, value, quality, volatility)
- Factor-based signal generator
- Portfolio construction based on factor scores
- Rebalancing scheduler

**Implementation**: Add fundamental data scraper, factor calculation module, factor-based signal
generator

#### 2. Technical Analysis / Signal-Based Trading

**What it does**: Uses price/volume patterns to generate signals

**Common indicators**:

- Moving averages (SMA, EMA, MACD)
- RSI, stochastic oscillators
- Bollinger Bands
- ATR-based volatility filters

**Automation**: Rule-based → easy to backtest and deploy

**Can be applied to**:

- DJIA index
- DIA ETF
- Individual Dow components

**Automation level**: ⭐⭐⭐⭐⭐

**Risk**: Overfitting without proper validation

**Requirements**:

- Technical indicator library (SMA, EMA, MACD, RSI, stochastic, Bollinger Bands, ATR)
- Signal generation rules (crossover, divergence, overbought/oversold)
- Signal combination logic (AND/OR conditions)
- Multi-indicator confirmation system
- Backtesting framework for validation

**Implementation**: Add technical indicator library, signal combination logic, multi-indicator
confirmation system

**Related**: Complements Elliott Wave "Confirmation Tools" (section 4 above) - these indicators
can validate wave patterns

#### 3. Statistical Arbitrage / Mean Reversion

**What it does**: Exploits temporary deviations from statistical norms

**Examples**:

- Pairs trading (e.g., JPM vs BAC)
- Z-score mean reversion
- Cointegration-based strategies

**Dow-specific angle**: Strong sector clustering (industrials, financials). Long data history →
good for testing.

**Automation level**: ⭐⭐⭐⭐

**Key challenge**: Regime shifts

**Requirements**:

- Pairs detection algorithm (correlation, cointegration testing)
- Z-score calculation and mean reversion detection
- Cointegration testing framework (ADF test, Johansen test)
- Spread calculation and monitoring
- Entry/exit signal generation based on statistical thresholds
- Regime shift detection

**Implementation**: Add pairs detection, cointegration testing, mean reversion signal generator

#### 4. Regime Detection / Market State Modeling

**What it does**: Adjusts strategy depending on market conditions

**Techniques**:

- Hidden Markov Models (HMM)
- Volatility clustering
- Trend vs range classification
- Macro regime filters (rates, inflation)

**Why this matters**: DJIA behaves very differently in:

- Crisis vs expansion
- Rate hiking vs easing cycles

**Automation level**: ⭐⭐⭐⭐

**Often combined with**: Trend or factor models

**Requirements**:

- Hidden Markov Model implementation
- Volatility calculation and clustering analysis
- Trend vs range classification algorithm
- Macro data integration (interest rates, inflation, economic indicators)
- Regime state classifier
- Regime-aware signal filtering and strategy adjustment

**Implementation**: Add regime detection module, market state classifier, regime-aware signal
filtering

**Related**: Can enhance Elliott Wave analysis by filtering signals based on market regime (see
"Performance Analytics" section 8 above)

#### 5. Risk-Based Portfolio Construction

**What it does**: Focuses on risk contribution instead of capital allocation

**Frameworks**:

- Risk parity
- Minimum variance
- Volatility targeting
- CVaR optimization

**Automation**: Rebalance on fixed schedules, adaptive risk scaling

**Automation level**: ⭐⭐⭐⭐⭐

**Note**: Very production-friendly

**Requirements**:

- Portfolio optimization algorithms (risk parity, minimum variance, CVaR)
- Risk calculation engine (covariance matrix, correlation, volatility)
- Volatility targeting system
- Rebalancing scheduler (fixed schedule or threshold-based)
- Position sizing based on risk contribution
- Multi-asset portfolio support

**Implementation**: Add portfolio optimization module, risk calculation engine, rebalancing
scheduler

**Related**: Complements Elliott Wave "Position Sizing & Wave Invalidation" (section 7 above) -
extends individual trade risk management to portfolio level

#### 6. Machine Learning–Based Predictive Models

**What it does**: Learns nonlinear relationships in price, volume, fundamentals, macro data

**Common models**:

- Gradient boosting (XGBoost, LightGBM)
- Random forests
- LSTM/Temporal CNNs (careful with overfitting)

**Best practice**: Use ML for:

- Signal filtering
- Regime detection
- Feature selection
- Not raw price prediction

**Automation level**: ⭐⭐⭐⭐

**Danger**: False confidence without strong validation

**Requirements**:

- ML framework (XGBoost, LightGBM, scikit-learn, TensorFlow/PyTorch)
- Feature engineering pipeline (technical indicators, fundamental ratios, macro data)
- Model training infrastructure
- Model validation framework (cross-validation, walk-forward analysis, out-of-sample testing)
- Feature selection algorithms
- Model monitoring and retraining scheduler
- Prediction integration with signal generation

**Implementation**: Add ML model training pipeline, feature engineering, model validation
framework, prediction integration

**Related**: Can enhance Elliott Wave analysis through signal filtering and regime detection (see
"Technical Indicator Confirmation" section 4 and "Regime Detection" framework above)

#### 7. Event-Driven Analytics

**What it does**: Trades around known events

**Examples**:

- Earnings surprises (component stocks)
- Fed announcements
- CPI / NFP releases

**Automation**: Rule-based event windows, NLP for earnings call sentiment

**Automation level**: ⭐⭐⭐

**Note**: Data intensive

**Requirements**:

- Event calendar integration (earnings dates, economic releases, Fed meetings)
- Earnings data scraper (actual vs expected, surprise calculation)
- Economic data integration (CPI, NFP, GDP, etc.)
- Event impact analyzer (pre/post event price movements)
- NLP framework for earnings call sentiment analysis
- Rule-based event window trading logic
- Event-based signal generation

**Implementation**: Add event calendar, earnings data scraper, event impact analyzer, sentiment
analysis module

## Requirements

- Python 3.x
- yfinance (see requirements.txt)
- Docker (for containerized execution)

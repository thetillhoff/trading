# Trade Evaluator

Evaluates trading signals by analyzing actual market outcomes. For each signal, determines
if the target price or stop-loss was hit first, and calculates percentage gains/losses.

## Overview

This tool evaluates the performance of trading signals generated from Elliott Wave patterns:

- **Target Hit**: Price reached the target before stop-loss → Winning trade
- **Stop-Loss Hit**: Price hit stop-loss before target → Losing trade
- **No Outcome**: Neither target nor stop-loss hit (data ended or max days reached)
- **Percentage Gains**: Calculates actual gain/loss percentage for each trade

## Features

- Evaluates buy and sell signals against historical market data
- Determines if targets or stop-losses were hit first
- Calculates percentage gains/losses for each trade
- Tracks maximum favorable and adverse price movements
- Provides summary statistics (win rate, average gain/loss, best/worst trades)
- **Visual chart showing all trades with color-coded wins/losses**
- Configurable maximum holding period
- Detailed trade-by-trade analysis

## How Trade Evaluation Works

### Buy Signals (Betting on Price Going Up)

1. **Entry**: Signal generated at end of Wave 2 or Wave 4 (correction in uptrend)
2. **Target**: Fibonacci extension level (typically 1.618x the correction size) - price going UP
3. **Stop-Loss**: Based on risk/reward ratio (e.g., if target is +10%, stop-loss might be -5% for 2:1 ratio) - price going DOWN
4. **Evaluation**: Check if price reached target (win) or stop-loss (loss) first
5. **Win**: Price went up and hit target → "Buy Win"
6. **Loss**: Price went down and hit stop-loss → "Buy Loss"

### Sell Signals (Betting on Price Going Down)

1. **Entry**: Signal generated at end of Wave 5 (impulse complete) or Wave B (counter-trend)
2. **Target**: Fibonacci retracement level (typically 50% of the impulse) - price going DOWN
3. **Stop-Loss**: Based on risk/reward ratio (above entry for sell signals) - price going UP
4. **Evaluation**: Check if price reached target (win) or stop-loss (loss) first
5. **Win**: Price went down and hit target → "Sell Win" (successful bearish bet)
6. **Loss**: Price went up and hit stop-loss → "Sell Loss" (bearish bet failed)

### Metrics Calculated

- **Gain Percentage**: `((exit_price - entry_price) / entry_price) * 100` for buys
- **Days Held**: Number of days from entry to exit
- **Max Favorable Excursion**: Best price movement in your favor during the trade
- **Max Adverse Excursion**: Worst price movement against you during the trade

## Requirements

- Docker (recommended)
- Or Python 3.11+ with dependencies from `requirements.txt`
- DJIA scraper data (run the scraper first)
- By default, all signals are evaluated (use `--require-both-targets` to restrict to signals with both target and stop-loss)

## Usage

### Using Docker (Recommended)

```bash
# Build the Docker image
make build

# Basic trade evaluation
make run ARGS="--column Close"

# With date range (last 10 years)
make run ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"

# Evaluate only buy signals with 30-day limit
make run ARGS="--column Close --signal-type buy --max-days 30"

# Custom risk/reward ratio
make run ARGS="--column Close --risk-reward 3.0"
```

### Using Docker directly

```bash
# Build
cd visualizations/trade_evaluator
docker build -t trade-evaluator .

# Run
docker run --rm \
  -v $(pwd)/../../scrapers:/app/scrapers:ro \
  -v $(pwd)/../djia:/app/djia:ro \
  -v $(pwd)/../trading_signals:/app/trading_signals:ro \
  -v $(pwd)/../../:/app/project_root \
  trade-evaluator python evaluate_trades.py [ARGS]
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python evaluate_trades.py [ARGS]
```

## Command Line Arguments

### Data Selection

- `--start-date YYYY-MM-DD`: Start date (inclusive)
- `--end-date YYYY-MM-DD`: End date (inclusive)
- `--column {Close|High|Low|Open|Volume}`: Column to analyze (default: Close)

### Signal Detection

- `--min-confidence FLOAT`: Minimum confidence for wave detection (default: 0.6)
- `--min-wave-size FLOAT`: Minimum wave size as ratio of price range (default: 0.05)
- `--signal-type {buy|sell|all}`: Type of signals to evaluate (default: all)

### Target Calculation

- `--risk-reward FLOAT`: Risk/reward ratio for stop-loss (default: 2.0)

### Evaluation Settings

- `--max-days INT`: Maximum days to hold a trade (default: until target/stop-loss or data ends)
- `--require-both-targets`: Only evaluate signals with both target and stop-loss (default: evaluate all signals)
- `--hold-through-stop-loss`: When stop-loss is hit, continue holding until recovery (price returns to entry level or better). Useful for testing "hold and wait" strategies vs. strict stop-loss execution.

### Output

- `--output-dir PATH`: Output directory for charts (default: current directory)
- `--output-filename NAME`: Custom output filename (auto-generated if not specified)
- `--title TITLE`: Custom chart title (auto-generated if not specified)
- `--no-chart`: Skip chart generation (only show text output)

## Examples

### Basic Evaluation

```bash
make run ARGS="--column Close"
```

Shows all trades with their outcomes and percentage gains.

### Focused Analysis

```bash
make run ARGS="--start-date 2020-01-01 --end-date 2024-12-31 --column Close --signal-type buy"
```

Evaluates only buy signals from 2020-2024.

### Short-Term Trading

```bash
make run ARGS="--column Close --max-days 30 --risk-reward 2.0"
```

Evaluates trades with a 30-day maximum holding period.

### Hold Through Stop-Loss (Test Recovery Strategy)

```bash
make run ARGS="--column Close --hold-through-stop-loss"
```

Simulates what happens if you don't exit when stop-loss is hit, but instead hold until price recovers (returns to entry level or better). Useful for comparing strict stop-loss execution vs. holding strategies.

## Visualization

The trade evaluator automatically generates a visual chart showing:

- **Price line**: Historical price data
- **Entry points**: Marked with triangles (^=buy, v=sell)
  - Green triangles: Winning trades (target hit)
  - Red triangles: Losing trades (stop-loss hit)
  - Orange triangles: No outcome (neither target nor stop-loss hit)
- **Exit points**: 
  - Green circles: Target price reached
  - Red X markers: Stop-loss hit
- **Connection lines**: Dashed lines connecting entry to exit points (shown in legend as "Entry to Exit Connection")
- **Annotations**: Percentage gain/loss shown near each exit point
- **Summary box**: Statistics displayed at the bottom of the chart (includes buy-and-hold comparison)

The chart uses the same format and style as the Elliott Wave and Trading Signals visualizations, making it easy to compare all three side-by-side.

## Output Interpretation

### Summary Statistics

- **Total Trades**: Number of signals evaluated
- **Winning Trades**: Trades where target was hit before stop-loss
- **Losing Trades**: Trades where stop-loss was hit before target
- **Win Rate**: Percentage of trades that were winners
- **Average Gain**: Average percentage gain for winning trades
- **Average Loss**: Average percentage loss for losing trades
- **Total Gain/Loss**: Sum of all percentage gains/losses
- **Buy-and-Hold Gain**: Gain if you simply bought at start and held until end
- **Outperformance**: Difference between trading strategy and buy-and-hold performance

### Trade Details

For each trade, you'll see:

- Entry date and price
- Target and stop-loss levels
- Outcome (target hit, stop-loss hit, or no outcome)
- Exit price and date
- Gain/loss percentage
- Days held
- Maximum favorable and adverse price movements

## Understanding the Results

### Win Rate

A win rate above 50% with a good risk/reward ratio (e.g., 2:1) indicates profitable strategy.
However, consider:
- Sample size (more trades = more reliable)
- Market conditions during the period
- Whether results are consistent across different time periods

### Average Gain vs Loss

For a 2:1 risk/reward ratio:
- If you win 40% of trades, you break even
- If you win 50% of trades, you're profitable
- Average gain should be roughly 2x the average loss

### Max Excursions

- **Max Favorable**: Shows how much better the trade could have been if you held longer
- **Max Adverse**: Shows how much worse it got before recovering (or hitting stop-loss)

### Buy-and-Hold Comparison

The evaluator automatically compares your trading strategy's performance against a simple buy-and-hold strategy:
- **Buy-and-Hold Gain**: What you would have made if you bought at the start of the period and held until the end
- **Outperformance**: The difference between your strategy's total gain and buy-and-hold gain
- Positive outperformance means your trading strategy beat buy-and-hold
- Negative outperformance means buy-and-hold would have been better

## Limitations

- **Hindsight Bias**: Results are based on historical data - past performance doesn't guarantee future results
- **Data Quality**: Results depend on accurate price data and signal detection
- **Market Conditions**: Results may vary significantly in different market environments
- **Execution**: Assumes you can enter/exit at exact target/stop-loss prices (real trading has slippage)

## Future Enhancements

Potential improvements:
- Portfolio-level analysis (multiple trades simultaneously)
- Risk-adjusted returns (Sharpe ratio, etc.)
- Drawdown analysis
- Trade correlation analysis
- Performance by market conditions (bull/bear markets)

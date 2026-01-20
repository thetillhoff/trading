# Trading Signals Analyzer

Analyzes Elliott Wave patterns to identify buy and sell signals, and calculates
target prices and stop-loss levels for trading decisions.

## Overview

This tool detects trading opportunities based on Elliott Wave Theory:
- **Buy Signals**: End of wave 2 or wave 4 (corrections in uptrend)
- **Sell Signals**: End of wave 5 (completion of impulse) or wave B (counter-trend)
- **Target Prices**: Calculated using Fibonacci extensions/retracements
- **Stop Loss**: Based on risk/reward ratio

## Features

- Automatic buy/sell signal detection from Elliott Wave patterns
- Target price calculation using Fibonacci levels
- Stop-loss calculation based on risk/reward ratio
- Visual markers on price charts (buy/sell points, targets, stop-loss)
- Filter by signal type (buy, sell, or all)
- Date range filtering
- Configurable confidence and wave size thresholds

## Requirements

- Docker (recommended)
- Or Python 3.11+ with dependencies from `requirements.txt`
- DJIA scraper data (run the scraper first to generate data)

## Usage

### Using Docker (Recommended)

```bash
# Build the Docker image
make build

# Basic signal analysis
make run ARGS="--column Close"

# With date range (last 10 years)
make run ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"

# Show only buy signals with custom risk/reward
make run ARGS="--column Close --signal-type buy --risk-reward 3.0"

# Custom confidence threshold
make run ARGS="--column Close --min-confidence 0.7 --min-wave-size 0.08"
```

### Using Docker directly

```bash
# Build
cd visualizations/trading_signals
docker build -t trading-signals-analyzer .

# Run
docker run --rm \
  -v $(pwd)/../../scrapers:/app/scrapers:ro \
  -v $(pwd)/../djia:/app/djia:ro \
  -v $(pwd)/../../:/app/project_root \
  trading-signals-analyzer python analyze_signals.py [ARGS] --output-dir /app/project_root
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python analyze_signals.py [ARGS]
```

## Command Line Arguments

### Data Selection

- `--start-date YYYY-MM-DD`: Start date (inclusive)
- `--end-date YYYY-MM-DD`: End date (inclusive)
- `--column {Close|High|Low|Open|Volume}`: Column to analyze (default: Close)

### Signal Detection

- `--min-confidence FLOAT`: Minimum confidence for wave detection (default: 0.6)
- `--min-wave-size FLOAT`: Minimum wave size as ratio of price range (default: 0.05)
- `--signal-type {buy|sell|all}`: Type of signals to show (default: all)

### Target Calculation

- `--risk-reward FLOAT`: Risk/reward ratio for stop-loss (default: 2.0)

### Output

- `--output-dir PATH`: Output directory for charts (default: current directory)
- `--output-filename NAME`: Custom output filename (auto-generated if not specified)
- `--title TITLE`: Custom chart title (auto-generated if not specified)

## Examples

### Basic analysis

```bash
make run ARGS="--column Close"
```

### Last 10 years with buy signals only

```bash
make run ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --signal-type buy"
```

### Conservative signals (higher confidence)

```bash
make run ARGS="--column Close --min-confidence 0.7 --min-wave-size 0.08"
```

### Aggressive risk/reward (3:1)

```bash
make run ARGS="--column Close --risk-reward 3.0"
```

## Output

The analyzer provides:

1. **Signal Summary**: Count of buy/sell signals detected
2. **Signal Details**: For each signal:
   - Date and price
   - Confidence level
   - Target price (with percentage change)
   - Stop-loss level (with risk percentage)
   - Reasoning for the signal

3. **Visualization**: Chart with:
   - Price line
   - Buy signals (green triangles)
   - Sell signals (red triangles)
   - Target prices (circles with dashed lines)
   - Stop-loss levels (X markers)

## Signal Types

### Buy Signals

- **End of Wave 2**: Correction in uptrend, potential entry before wave 3
- **End of Wave 4**: Correction in uptrend, potential entry before wave 5
- **End of Wave B (down)**: Corrective wave down in uptrend

### Sell Signals

- **End of Wave 5**: Completion of impulse wave, potential exit
- **End of Wave B (up)**: Corrective wave up in downtrend

## Target Calculation

Targets are calculated using:
- **Fibonacci extensions** (for buy targets): 1.618x the correction size
- **Fibonacci retracements** (for sell targets): 50% of the impulse size
- **Stop-loss**: Based on risk/reward ratio (default 2:1)

## Integration with Other Tools

This tool works well with:
- **Elliott Wave Optimizer**: Use optimized filter values for better signal detection
- **Visualization**: Signals can be overlaid on existing charts

## Future Enhancements

This module is designed to grow. Potential future additions:

- Multiple timeframe analysis
- Signal strength scoring
- Backtesting capabilities
- Portfolio-level signal aggregation
- Alert system for new signals
- Custom target calculation strategies
- Integration with trading platforms

## Troubleshooting

- **No signals detected**: Try lowering `--min-confidence` or `--min-wave-size`
- **Too many signals**: Increase thresholds or use date filtering
- **Insufficient data**: Ensure you have enough historical data (100+ days recommended)
- **Docker errors**: Ensure Docker is running and paths are correct

# Elliott Wave Filter Optimizer

Finds optimal filter parameters for Elliott Wave detection by analyzing patterns
across different time granularities (yearly, quarterly, monthly, weekly, daily).

## Overview

This tool analyzes your trading data to recommend optimal values for:

- `--min-confidence`: Minimum confidence threshold for wave detection
- `--min-wave-size`: Minimum wave size as ratio of price range
- `--only-complete-patterns`: Whether to only show complete patterns

The optimizer examines data at multiple time granularities to find the best filter
values that balance visual clarity with meaningful pattern detection.

## Features

- **Multi-granularity analysis**: Analyzes yearly, quarterly, monthly, weekly, and
  daily patterns
- **Automatic optimization**: Finds optimal filter values based on data characteristics
- **Target wave count**: Can optimize for a specific number of waves to display
- **Filter testing**: Test specific filter combinations to see their impact
- **Detailed reporting**: Shows analysis results for all granularities

## Requirements

- Docker (recommended)
- Or Python 3.11+ with dependencies from `requirements.txt`
- DJIA scraper data (run the scraper first to generate data)

## Usage

### Using Docker (Recommended)

```bash
# Build the Docker image
make build

# Auto-optimize filters (recommended)
make run

# Optimize for specific wave count
make run ARGS="--target-waves 10"

# Analyze specific granularity
make run ARGS="--granularity monthly"

# Show detailed analysis
make run ARGS="--verbose"

# Test specific filter values
make run ARGS="--test-filters 'confidence=0.7,size=0.1'"
```

### Using Docker directly

```bash
# Build
cd visualizations/elliott_wave_optimizer
docker build -t elliott-wave-optimizer .

# Run
docker run --rm \
  -v $(pwd)/../../scrapers:/app/scrapers:ro \
  -v $(pwd)/../../djia:/app/djia:ro \
  elliott-wave-optimizer python optimize_filters.py [ARGS]
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimizer
python optimize_filters.py [ARGS]
```

## Command Line Arguments

### Optimization

- `--target-waves N`: Target number of waves to display (auto if not specified)
- `--granularity {yearly|quarterly|monthly|weekly|daily}`: Preferred granularity
  for analysis (auto-selects best if not specified)

### Analysis

- `--verbose`: Show detailed analysis for all granularities
- `--test-filters "confidence=X,size=Y"`: Test specific filter values

## Examples

### Basic optimization

```bash
make run
```

This will analyze all granularities and recommend optimal filter values.

### Optimize for specific wave count

```bash
make run ARGS="--target-waves 15"
```

Finds filters that will show approximately 15 waves.

### Analyze monthly patterns

```bash
make run ARGS="--granularity monthly --verbose"
```

Focuses on monthly granularity and shows detailed statistics.

### Test filter impact

```bash
make run ARGS="--test-filters 'confidence=0.7,size=0.1'"
```

Shows how many waves would be detected with these specific filters.

## Output

The optimizer provides:

1. **Analysis Results**: Statistics for each granularity:
   - Data points
   - Waves detected
   - Average confidence
   - Wave size distribution
   - Recommended filter values

2. **Recommended Filters**: Optimal values to use with visualization:
   - `--min-confidence`
   - `--min-wave-size`
   - `--only-complete-patterns` (if applicable)

3. **Usage Command**: Ready-to-use command with recommended filters

## Integration with Visualization

After running the optimizer, use the recommended values:

```bash
# Get recommendations
make run

# Use recommended values (example)
make visualize ARGS="--granularity daily --column Close --elliott-waves \
  --min-confidence 0.7 --min-wave-size 0.08"
```

## Future Enhancements

This module is designed to grow. Potential future additions:

- Machine learning-based optimization
- Historical pattern matching
- Multi-timeframe analysis
- Pattern quality scoring
- Custom optimization strategies
- Visualization of optimization results
- Automated filter adjustment over time

## Troubleshooting

- **No waves detected**: Try lowering `--min-confidence` or `--min-wave-size`
- **Too many waves**: Use recommended filters or increase thresholds
- **Insufficient data**: Ensure you have enough historical data (100+ days recommended)
- **Docker errors**: Ensure Docker is running and paths are correct

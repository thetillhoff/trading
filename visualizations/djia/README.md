# DJIA Visualization

Generates customizable charts and graphs from DJIA trading data.

## Overview

This visualization tool creates line charts from DJIA data with support for:

- Time range filtering
- Multiple granularities (daily, weekly, monthly, yearly)
- Various aggregation methods (mean, max, min, median, sum, first, last)
- Column selection (Close, High, Low, Open, Volume)
- Elliott Wave detection and color coding

## Features

- Flexible time range selection
- Multiple time granularities
- Various aggregation methods
- Column selection
- Elliott Wave pattern detection and color coding
- High-quality PNG output (300 DPI)
- Automatic data loading from scraper outputs

## Requirements

- Docker (recommended)
- Or Python 3.11+ with dependencies from `requirements.txt`
- DJIA scraper data (run the scraper first to generate data)

## Usage

### Using Docker (Recommended)

```bash
# Build the Docker image
make build

# Generate a visualization
make run ARGS="--granularity daily --column Close"

# Or build and run in one command
make build && make run ARGS="--granularity monthly --aggregation mean --column Close"
```

### Using Docker directly

```bash
# Build
cd visualizations/djia
docker build -t djia-visualizer .

# Run (mounts scraper data and project root for output)
docker run --rm \
  -v $(pwd)/../../scrapers:/app/scrapers:ro \
  -v $(pwd)/../../:/app/project_root \
  djia-visualizer python visualize_djia.py [ARGS]
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run visualization
python visualize_djia.py [ARGS]
```

## Command Line Arguments

### Time Range

- `--start-date YYYY-MM-DD`: Start date (inclusive)
- `--end-date YYYY-MM-DD`: End date (inclusive)

### Granularity

- `--granularity {daily|weekly|monthly|yearly}`: Time granularity (default: daily)

### Aggregation

- `--aggregation {mean|max|min|median|sum|first|last}`: Aggregation method (default: mean)
- Only applies when granularity is not daily

### Column Selection

- `--column {Close|High|Low|Open|Volume}`: Column to visualize (default: Close)

### Output

- `--output-dir PATH`: Output directory for charts (default: project root)
- `--output-filename NAME`: Custom output filename (auto-generated if not specified)
- `--title TITLE`: Custom chart title (auto-generated if not specified)

### Elliott Waves

- `--elliott-waves`: Enable Elliott Wave detection and color coding on the chart
- `--min-confidence FLOAT`: Minimum confidence (0.0-1.0) for wave detection (default: 0.6).
  Higher values show only more confident detections, reducing visual clutter.
- `--min-wave-size FLOAT`: Minimum wave size as ratio of price range (default: 0.05 = 5%).
  **Note**: This is a practical filter to reduce noise, not a requirement from Elliott Wave
  theory. Elliott Wave theory focuses on wave relationships and Fibonacci ratios, not absolute
  size thresholds. Use the filter optimizer to find appropriate values for your data.
  Higher values filter out smaller waves. For example, 0.1 = 10% of price range.
- `--only-complete-patterns`: Only show complete 5-wave impulse or 3-wave corrective patterns.
  This significantly reduces the number of waves displayed.

## Examples

### Daily close prices for 2023

```bash
make run ARGS="--start-date 2023-01-01 --end-date 2023-12-31 --granularity daily --column Close"
```

### Monthly average prices (all time)

```bash
make run ARGS="--granularity monthly --aggregation mean --column Close"
```

### Weekly maximum high prices

```bash
make run ARGS="--granularity weekly --aggregation max --column High"
```

### Yearly median prices for a specific range

```bash
make run ARGS="--start-date 2020-01-01 --end-date 2023-12-31 --granularity yearly --aggregation median --column Close"
```

### Custom output location and title

```bash
make run ARGS="--granularity daily --column Close --output-dir ./charts --title 'DJIA Close Prices 2023'"
```

### Daily prices with Elliott Wave detection

```bash
make run ARGS="--granularity daily --column Close --elliott-waves"
```

### Elliott Waves for a specific date range

```bash
make run ARGS="--start-date 2023-01-01 --end-date 2023-12-31 --granularity daily --column Close --elliott-waves"
```

### Filtered Elliott Waves (reduce visual clutter)

```bash
# Only show high-confidence waves (0.7+ confidence)
make run ARGS="--granularity daily --column Close --elliott-waves --min-confidence 0.7"

# Only show significant waves (10%+ of price range)
make run ARGS="--granularity daily --column Close --elliott-waves --min-wave-size 0.1"

# Only show complete patterns (most restrictive)
make run ARGS="--granularity daily --column Close --elliott-waves --only-complete-patterns"

# Combine filters for cleaner visualization
make run ARGS="--granularity daily --column Close --elliott-waves --min-confidence 0.7 --min-wave-size 0.08"
```

## Output Files

Charts are saved as PNG files (300 DPI) in the specified output directory (or project root by default).
Filenames are auto-generated based on parameters, or you can specify a custom filename.

## Data Source

The visualization automatically loads data from the DJIA scraper output:

- Looks for `scrapers/djia/djia_data.csv`
- Works in both local development and Docker environments
- Data is read-only (visualization doesn't modify scraper data)

## Troubleshooting

- **Data file not found**: Run the DJIA scraper first to generate the data file
- **Column not found**: Check available columns with `--column` help or inspect the data file
- **Date parsing errors**: Ensure dates are in YYYY-MM-DD format
- **Docker volume mount errors**: Ensure paths are correct and Docker has permissions

## Advanced Usage

### Multiple visualizations in a script

```bash
# Generate multiple charts
make run ARGS="--granularity daily --column Close --start-date 2023-01-01 --end-date 2023-03-31"
make run ARGS="--granularity daily --column Close --start-date 2023-04-01 --end-date 2023-06-30"
make run ARGS="--granularity daily --column Close --start-date 2023-07-01 --end-date 2023-09-30"
make run ARGS="--granularity daily --column Close --start-date 2023-10-01 --end-date 2023-12-31"
```

### Combining with other tools

The generated PNG files can be used with:

- Image viewers
- Documentation tools
- Report generators
- Web applications

## Understanding Elliott Wave Patterns

### What Are Elliott Waves?

Elliott Wave Theory is a technical analysis method that identifies recurring patterns in
price movements. The theory suggests that market prices move in predictable wave patterns
that reflect investor psychology.

### Wave Structure

**Impulse Waves (1-5)**: These move in the direction of the main trend:

- **Wave 1**: Initial move in the trend direction
- **Wave 2**: Correction/retracement (typically 38.2% to 61.8% of Wave 1)
- **Wave 3**: Strongest wave, often the longest (cannot be shortest of 1, 3, 5)
- **Wave 4**: Another correction (typically 23.6% to 38.2% of Wave 3)
- **Wave 5**: Final move in trend direction (often weaker than Wave 3)

**Corrective Waves (a, b, c)**: These move against the main trend:

- **Wave A**: First correction against the trend
- **Wave B**: Partial retracement of Wave A
- **Wave C**: Final correction, often extends beyond Wave A

### Color Coding

Each wave type is displayed with a distinct color for easy visual identification.
The color scheme is defined in the visualization code (see `visualizer.py`) and ensures
all 8 waves (1-5, a-c) are clearly distinguishable. Impulse waves (1-5) and corrective
waves (a-c) use different color families to help distinguish wave types at a glance.

### Price Forecast Interpretation

#### Impulse Wave Patterns (1-2-3-4-5)

When you see a complete 5-wave impulse pattern:

1. **During Wave 2 or 4**: These are potential **buy opportunities** in an uptrend
   (or sell opportunities in a downtrend). The correction is expected to end, and
   the next impulse wave should follow.

2. **End of Wave 5**: This suggests the trend may be **exhausted**. After a 5-wave
   impulse, expect a larger correction (typically a 3-wave a-b-c pattern).

3. **Wave 3 Strength**: If Wave 3 is particularly strong and long, it indicates
   strong momentum in the trend direction.

#### Corrective Wave Patterns (a-b-c)

When you see a 3-wave corrective pattern:

1. **End of Wave C**: This often marks the **end of a correction**. After a
   complete a-b-c correction, the original trend typically resumes.

2. **Wave B Retracement**: Wave B typically retraces 38.2% to 78.6% of Wave A.
   A shallow Wave B suggests strong underlying trend.

### Forecasting Guidelines

**Uptrend Pattern (5 waves up)**:

- After Wave 5 completes → Expect downward correction (a-b-c)
- After correction completes → Potential for new uptrend or continuation

**Downtrend Pattern (5 waves down)**:

- After Wave 5 completes → Expect upward correction (a-b-c)
- After correction completes → Potential for new downtrend or reversal

**Key Rules**:

- Wave 2 cannot retrace more than 100% of Wave 1
- Wave 3 cannot be the shortest of waves 1, 3, and 5
- Wave 4 cannot overlap with Wave 1 (except in diagonal triangles)
- Wave 3 is often the longest and strongest wave

### Limitations

**Important**: Elliott Wave analysis is:

- **Subjective**: Different analysts may identify different wave counts
- **Probabilistic**: Patterns suggest probabilities, not certainties
- **Context-dependent**: Works best with sufficient historical data
- **Not a guarantee**: Past patterns don't guarantee future results

Use Elliott Wave analysis as **one tool** in your trading toolkit, combined with:

- Fundamental analysis
- Other technical indicators
- Risk management
- Market context

### Best Practices

1. **Use with date filtering**: Focus on recent data (e.g., last 10 years) for
   more relevant patterns
2. **Combine with filter optimization**: Use the optimizer to find appropriate
   filter values for your data
3. **Look for complete patterns**: 5-wave impulses or 3-wave corrections are
   more reliable than incomplete patterns
4. **Consider multiple timeframes**: Patterns at different granularities can
   provide context
5. **Use with trading signals**: The trading signals module can help identify
   specific entry/exit points based on wave patterns

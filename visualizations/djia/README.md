# DJIA Visualization

Generates customizable charts and graphs from DJIA trading data.

## Overview

This visualization tool creates line charts from DJIA data with support for:

- Time range filtering
- Multiple granularities (daily, weekly, monthly, yearly)
- Various aggregation methods (mean, max, min, median, sum, first, last)
- Column selection (Close, High, Low, Open, Volume)

## Features

- Flexible time range selection
- Multiple time granularities
- Various aggregation methods
- Column selection
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

# Trading Analysis - Best Practices & Examples

This document contains recommended commands and best practices for using the trading analysis tools.

## Trade Evaluation

### Recommended: Full Evaluation with All Signals

**Best for**: Getting comprehensive trade analysis without filtering out signals that lack targets/stop-loss.

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"
```

**Why this is recommended**:

- Evaluates all detected signals by default (most comprehensive analysis)
- Provides complete view of all trading opportunities
- Still calculates targets/stop-loss where possible
- Generates visualization showing all evaluated trades
- Use `--require-both-targets` if you only want signals with complete target/stop-loss

**Output includes**:

- Total trades evaluated
- Win rate and performance statistics
- Buy-and-hold comparison
- Visual chart with all trades marked

### Other Useful Variations


**Stricter filtering (fewer, higher-quality signals)**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --min-confidence 0.7 --min-wave-size 0.08"
```

**Only buy signals**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --signal-type buy"
```

**Short-term trading (30-day max hold)**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --max-days 30"
```

**Higher risk/reward ratio**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --risk-reward 3.0"
```

**Hold through stop-loss (test recovery strategy)**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --hold-through-stop-loss"
```

**Only signals with both target and stop-loss**:

```bash
make evaluate-trades ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --require-both-targets"
```

This simulates what happens if you don't sell when stop-loss is hit, but instead hold until the price recovers (returns to entry level or better). Useful for comparing strict stop-loss execution vs. holding strategies.

## Multi-Chart Generation

### Combined Analysis Chart

Generate both Elliott Wave and Trading Signals in one image:

```bash
make multi-charts ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --combined"
```

### Separate Charts

Generate Elliott Wave and Trading Signals as separate images:

```bash
make multi-charts ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"
```

## Elliott Wave Visualization

### Standard Visualization

```bash
make visualize ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --elliott-waves"
```

### With Custom Filters

```bash
make visualize ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --elliott-waves --min-confidence 0.7 --min-wave-size 0.1"
```

## Filter Optimization

### Auto-Optimize Filters

```bash
make optimize-filters
```

### Optimize for Specific Wave Count

```bash
make optimize-filters ARGS="--target-waves 10"
```

### Optimize for Specific Time Period

```bash
make optimize-filters ARGS="--start-date 2020-01-01 --end-date 2024-12-31 --column Close"
```

## Trading Signals Analysis

### Basic Signal Detection

```bash
make trading-signals ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close"
```

### Only Buy Signals

```bash
make trading-signals ARGS="--start-date 2015-01-01 --end-date 2024-12-31 --column Close --signal-type buy"
```

## Tips

1. **Default behavior**: By default, all signals are evaluated (most comprehensive view)
2. **Use date ranges**: Focus on recent data (e.g., last 5-10 years) for more relevant patterns
3. **Compare time periods**: Run evaluations for different periods to see how strategy performs in different market conditions
4. **Use filter optimizer**: Before setting custom filters, run the optimizer to get data-driven recommendations
5. **Generate combined charts**: Use `--combined` flag to see Elliott Waves and Trading Signals side-by-side

#!/bin/bash
# Run all hypothesis tests systematically using parallel grid-search
#
# This script runs 37 hypothesis test configs across multiple time periods using
# parallel execution (8 workers by default) via Docker and grid-search.
#
# Usage:
#   ./scripts/run_hypothesis_tests.sh [category] [period]
#
# Examples:
#   ./scripts/run_hypothesis_tests.sh                      # Run all 37 configs on all 10 periods (~1-2 hours)
#   ./scripts/run_hypothesis_tests.sh rsi_tests            # Run only RSI tests (6 configs) on all periods
#   ./scripts/run_hypothesis_tests.sh rsi_tests quick_test # Run RSI tests on 2018-2020 only (~3 min)
#
# Available categories: rsi_tests, ema_tests, macd_tests, elliott_tests, combination_tests, regime_tests, baseline_tests
# Available periods: quick_test, recent_2yr, full_period_20yr, bear_market_long, bull_market_long, and more
#
# Results: CSV files in results/hypothesis_tests_YYYYMMDD_HHMMSS/[period]/
# Find best config: grep -h 'Alpha' results/hypothesis_tests_*/*/backtest_comparison_results.csv | sort -t',' -k3 -rn | head -10

set -e  # Exit on error

# Configuration
RESULTS_DIR="results/hypothesis_tests_$(date +%Y%m%d_%H%M%S)"
INSTRUMENT="djia"
WORKERS=8  # Adjust based on CPU cores

# Function to get period dates
get_period_dates() {
    case "$1" in
        quick_test) echo "2018-01-01 2020-01-01" ;;
        recent_2yr) echo "2020-01-01 2022-01-01" ;;
        covid_crash) echo "2019-01-01 2021-01-01" ;;
        recent_bull) echo "2015-01-01 2020-01-01" ;;
        recovery_period) echo "2009-01-01 2014-01-01" ;;
        housing_crisis) echo "2007-01-01 2010-01-01" ;;
        dotcom_crash) echo "2000-01-01 2003-01-01" ;;
        bear_market_long) echo "2000-01-01 2010-01-01" ;;
        bull_market_long) echo "2010-01-01 2020-01-01" ;;
        full_period_20yr) echo "2000-01-01 2020-01-01" ;;
        *) echo "" ;;
    esac
}

# All available periods
ALL_PERIODS="quick_test recent_2yr covid_crash recent_bull recovery_period housing_crisis dotcom_crash bear_market_long bull_market_long full_period_20yr"

# Priority periods (run these first)
PRIORITY_PERIODS="quick_test recent_2yr full_period_20yr"

# Test categories
CATEGORIES=(
    "rsi_tests"
    "ema_tests"
    "macd_tests"
    "elliott_tests"
    "combination_tests"
    "regime_tests"
    "baseline_tests"
)

# Parse arguments
SELECTED_CATEGORY="${1:-all}"
SELECTED_PERIOD="${2:-all}"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Hypothesis Testing Framework"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "Instrument: $INSTRUMENT"
echo ""

# Function to run parallel grid search for a period
run_period() {
    local period_name="$1"
    local start_date="$2"
    local end_date="$3"
    local config_dir="$4"
    
    local period_results_dir="${RESULTS_DIR}/${period_name}"
    mkdir -p "$period_results_dir"
    
    echo ""
    echo "========================================"
    echo "Period: ${period_name} (${start_date} to ${end_date})"
    echo "========================================"
    echo "Config directory: $config_dir"
    echo "Results directory: $period_results_dir"
    echo "Running with parallel execution (${WORKERS} workers)..."
    echo ""
    
    # Run grid-search with --config-dir and --parallel
    if make grid-search ARGS="--config-dir $config_dir --parallel --workers $WORKERS --start-date $start_date --end-date $end_date --instrument $INSTRUMENT --output-dir $period_results_dir --csv" \
        2>&1 | tee "${period_results_dir}/run.log"; then
        echo "✓ Period $period_name completed successfully"
    else
        echo "✗ Period $period_name failed (see ${period_results_dir}/run.log)"
    fi
}

# Determine which config directory to use
if [ "$SELECTED_CATEGORY" = "all" ]; then
    CONFIG_DIR="configs/hypothesis_tests"
else
    CONFIG_DIR="configs/hypothesis_tests/${SELECTED_CATEGORY}"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "Error: Category directory not found: $CONFIG_DIR"
        exit 1
    fi
fi

# Determine which periods to test
PERIODS_TO_TEST=""
if [ "$SELECTED_PERIOD" = "all" ]; then
    # Use priority periods first, then all others
    PERIODS_TO_TEST="$PRIORITY_PERIODS"
    for period in $ALL_PERIODS; do
        if ! echo "$PRIORITY_PERIODS" | grep -q "$period"; then
            PERIODS_TO_TEST="$PERIODS_TO_TEST $period"
        fi
    done
else
    PERIODS_TO_TEST="$SELECTED_PERIOD"
fi

# Run tests for each period
for period_name in $PERIODS_TO_TEST; do
    period_dates=$(get_period_dates "$period_name")
    if [ -z "$period_dates" ]; then
        echo "Warning: Unknown period: $period_name"
        continue
    fi
    start_date="${period_dates%% *}"
    end_date="${period_dates##* }"
    
    run_period "$period_name" "$start_date" "$end_date" "$CONFIG_DIR"
done

echo ""
echo "========================================"
echo "Testing Complete"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo "  - Each period has its own subdirectory with CSV results"
echo "  - Run logs saved as run.log in each period directory"
echo ""
echo "Next steps:"
echo "  1. Review CSV results in each period directory"
echo "  2. Compare results across periods"
echo ""
echo "To find best overall config:"
echo "  grep -h 'Alpha' $RESULTS_DIR/*/backtest_comparison_results.csv | sort -t',' -k3 -rn | head -10"
echo "========================================"

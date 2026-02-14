.PHONY: help clean download evaluate grid-search asset-analysis recommend test test-cov baseline-snapshot-generate

# Default target
help:
	@echo "Trading Strategy Backtesting System"
	@echo ""
	@echo "Core Commands:"
	@echo "  make download              Download/update market data for instruments"
	@echo "  make download-baseline     Download data for baseline config (run once before make evaluate)"
	@echo "  make evaluate              Evaluate a strategy (default: baseline.yaml, generates charts)"
	@echo "  make grid-search           Compare multiple configs from configs/ (auto-parallel, auto-charts)"
	@echo "  make recommend             Get today's best trade recommendation (baseline config)"
	@echo "  make hypothesis-tests      Run multi-period hypothesis suites (category/period selection)"
	@echo "  make asset-analysis        Run asset analysis (metadata, vol/correlation, candidate ranking; cache-first)"
	@echo "  make test                  Run tests with pytest"
	@echo "  make test-cov              Run tests with coverage report (development only)"
	@echo "  make baseline-snapshot-generate  Create/refresh baseline trades snapshot (2012; for regression test)"
	@echo ""
	@echo "Command Options (append to any command):"
	@echo "  ARGS='...'                 Pass CLI arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make download-baseline                                 # One-time: fetch baseline data (sp500)"
	@echo "  make download ARGS='--list'                           # List available instruments"
	@echo "  make download ARGS='sp500 djia'                       # Download specific data"
	@echo "  make evaluate                                           # Evaluate baseline.yaml (after download-baseline)"
	@echo "  make evaluate ARGS='--config configs/top_performers/ew_rsi.yaml'"
	@echo "  make recommend                                          # Get today's recommendation"
	@echo "  make recommend ARGS='--date 2026-01-15'                # Simulate recommendation for specific date"
	@echo "  make grid-search                                         # Test all configs in configs/"
	@echo "  make grid-search ARGS='--config-dir configs/optimization'  # Test specific subdirectory"
	@echo "  make grid-search ARGS='--output-dir results/my_run ...'    # Direct outputs to a path (e.g. for hypothesis-test flows)"
	@echo "  make grid-search ARGS='--analyze results/hypothesis_tests_YYYYMMDD/'  # Run analysis only on an existing results dir"
	@echo "  make asset-analysis ARGS='--fetch-metadata --analyze'  # Fetch metadata then run full analysis (cache-first)"
	@echo "  make asset-analysis ARGS='--all-assets --fetch-metadata --analyze'  # Metadata + OHLCV for all (cache-first), full ranking"
	@echo "  make asset-analysis ARGS='--all-assets --fetch-metadata --analyze --top 100'  # Same, write only top 100 to ranking CSVs"
	@echo "  make asset-analysis ARGS='--download-candidates'  # Download OHLCV for all in data/asset_analysis/candidate_ranking.csv"
	@echo "  make asset-analysis ARGS='--download-candidates --top 10'  # Download OHLCV for top 10 rows only"
	@echo ""
	@echo "Hypothesis tests (multi-period + analysis):"
	@echo "  make hypothesis-tests ARGS='--category rsi_tests --period quick_test'  # Run hypothesis suite via cli.hypothesis"
	@echo ""
	@echo "Data Directory:"
	@echo "  ./data/                    Downloaded market data (created by download command)"
	@echo "  ./data/tickers/             OHLCV for discovered tickers (asset-analysis --download-candidates)"
	@echo "  ./data/asset_analysis/      Asset analysis outputs (candidate_ranking, volatility, correlation; not strategy results)"
	@echo ""
	@echo "Results:"
	@echo "  ./results/                 Per-config results (matching configs/ structure)"
	@echo "  ./results/grid_search_*/    Grid search summary (charts, analysis)"
	@echo "  ./results/hypothesis_tests_*/  Multi-period runs from run_hypothesis_tests.sh; analysis_report.md written there after each run"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 Remove unused Docker images"

# Unified CLI (runs in Docker)
DOCKER_CLI = docker compose run --rm --build cli python -u -m

download:
	$(DOCKER_CLI) cli.download $(ARGS)

# Baseline uses sp500; download once so "make evaluate" works without args
download-baseline:
	$(DOCKER_CLI) cli.download sp500

evaluate:
	$(DOCKER_CLI) cli.evaluate $(ARGS)

grid-search:
	$(DOCKER_CLI) cli.grid_search $(ARGS)

hypothesis-tests:
	$(DOCKER_CLI) cli.hypothesis $(ARGS)

asset-analysis:
	$(DOCKER_CLI) cli.asset_analysis $(ARGS)

recommend:
	$(DOCKER_CLI) cli.recommend $(ARGS)

# Testing
test:
	docker compose run --rm --build cli pytest

test-cov:
	docker compose run --rm --build cli pytest --cov=core --cov-report=term-missing --cov-report=html

# Baseline snapshot: run after make download to create/refresh tests/snapshots/baseline_trades_short.csv
baseline-snapshot-generate:
	docker compose run --rm --build cli python scripts/generate_baseline_snapshot.py

# Cleanup
clean:
	docker system prune -f || true

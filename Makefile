.PHONY: help clean download evaluate grid-search test test-cov baseline-snapshot-generate hypothesis-long-short

# Default target
help:
	@echo "Trading Strategy Backtesting System"
	@echo ""
	@echo "Core Commands:"
	@echo "  make download              Download/update market data for instruments"
	@echo "  make evaluate              Evaluate a strategy (default: baseline.yaml, generates charts)"
	@echo "  make grid-search           Compare multiple configs from configs/ (auto-parallel, auto-charts)"
	@echo "  make hypothesis-tests      Run multi-period hypothesis suites (category/period selection)"
	@echo "  make test                  Run tests with pytest"
	@echo "  make test-cov              Run tests with coverage report (development only)"
	@echo "  make baseline-snapshot-generate  Create/refresh baseline trades snapshot (2012; for regression test)"
	@echo "  make hypothesis-long-short       Print long vs short trade breakdown from latest baseline trades CSV"
	@echo ""
	@echo "Command Options (append to any command):"
	@echo "  ARGS='...'                 Pass CLI arguments"
	@echo ""
	@echo "Examples:"
	@echo "  make download ARGS='--list'                           # List available instruments"
	@echo "  make download ARGS='djia sp500'                       # Download specific data"
	@echo "  make evaluate                                           # Evaluate baseline.yaml with charts"
	@echo "  make evaluate ARGS='--config configs/top_performers/ew_rsi.yaml'"
	@echo "  make grid-search                                         # Test all configs in configs/"
	@echo "  make grid-search ARGS='--config-dir configs/optimization'  # Test specific subdirectory"
	@echo "  make grid-search ARGS='--output-dir results/my_run ...'    # Direct outputs to a path (e.g. for hypothesis-test flows)"
	@echo "  make grid-search ARGS='--analyze results/hypothesis_tests_YYYYMMDD/'  # Run analysis only on an existing results dir"
	@echo ""
	@echo "Hypothesis tests (multi-period + analysis):"
	@echo "  make hypothesis-tests ARGS='--category rsi_tests --period quick_test'  # Run hypothesis suite via cli.hypothesis"
	@echo ""
	@echo "Data Directory:"
	@echo "  ./data/                    Downloaded market data (created by download command)"
	@echo ""
	@echo "Results:"
	@echo "  ./results/                 Per-config results (matching configs/ structure)"
	@echo "  ./results/grid_search_*/    Grid search summary (charts, analysis)"
	@echo "  ./results/hypothesis_tests_*/  Multi-period runs from run_hypothesis_tests.sh; analysis_report.md written there after each run"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 Remove unused Docker images"

# Pass PRODUCTION flag to sub-make
ifdef PRODUCTION
  PROD_FLAG = PRODUCTION=1
else
  PROD_FLAG =
endif

# Unified CLI (runs in Docker)
DOCKER_CLI = docker compose run --rm --build cli python -u -m

download:
	$(DOCKER_CLI) cli.download $(ARGS)

evaluate:
	$(DOCKER_CLI) cli.evaluate $(ARGS)

grid-search:
	$(DOCKER_CLI) cli.grid_search $(ARGS)

hypothesis-tests:
	$(DOCKER_CLI) cli.hypothesis $(ARGS)

# Testing
test:
	docker compose run --rm --build cli pytest

test-cov:
	docker compose run --rm --build cli pytest --cov=core --cov-report=term-missing --cov-report=html

# Baseline snapshot: run after make download to create/refresh tests/snapshots/baseline_trades_short.csv
baseline-snapshot-generate:
	docker compose run --rm --build cli python scripts/generate_baseline_snapshot.py

# Long vs short: analyze latest baseline trades CSV (run make evaluate first to produce trades)
hypothesis-long-short:
	docker compose run --rm --build cli python scripts/run_long_short_hypothesis.py

# Cleanup
clean:
	docker system prune -f || true

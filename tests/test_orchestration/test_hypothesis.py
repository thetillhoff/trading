import types
from pathlib import Path

import pytest

from core.orchestration.hypothesis import (
    HypothesisRunConfig,
    get_period_dates,
    resolve_periods,
    run_hypothesis_suite,
)


class TestPeriodHelpers:
    def test_get_period_dates_known_period(self):
        start, end = get_period_dates("quick_test")
        assert start == "2018-01-01"
        assert end == "2020-01-01"

    def test_get_period_dates_unknown_period(self):
        assert get_period_dates("unknown_period") is None

    def test_resolve_periods_all_uses_priorities_first(self):
        all_periods = ["a", "b", "c"]
        priority = ["b"]
        out = resolve_periods("all", all_periods, priority)
        assert out[0] == "b"
        # remaining periods appear once each
        assert set(out) == {"a", "b", "c"}

    def test_resolve_periods_explicit_list(self):
        all_periods = ["a", "b", "c"]
        priority = ["b"]
        out = resolve_periods(["c", "a"], all_periods, priority)
        assert out == ["c", "a"]


class TestRunHypothesisSuite:
    def test_run_hypothesis_suite_invokes_grid_search_and_analysis(self, monkeypatch, tmp_path):
        calls = []

        # Patch _run_grid_search_for_period to avoid heavy work
        import core.orchestration.hypothesis as hyp

        def fake_run_grid_search_for_period(**kwargs):
            calls.append(kwargs)
            # Simulate success
            return 0

        monkeypatch.setattr(hyp, "_run_grid_search_for_period", fake_run_grid_search_for_period)

        # Patch analyze_results_dir to record invocation
        analyzed = {}

        def fake_analyze_results_dir(results_dir, verbose=True):
            analyzed["dir"] = Path(results_dir)
            analyzed["verbose"] = verbose
            return True

        monkeypatch.setattr(hyp, "analyze_results_dir", fake_analyze_results_dir)

        cfg = HypothesisRunConfig(
            results_dir=tmp_path / "hypo",
            instrument="djia",
            workers=4,
            category="all",
            periods=["quick_test"],
            priority_periods=["quick_test"],
            all_periods=["quick_test"],
        )

        # Ensure configs/hypothesis_tests exists (minimal stub)
        config_dir = Path("configs") / "hypothesis_tests"
        config_dir.mkdir(parents=True, exist_ok=True)

        results_dir = run_hypothesis_suite(cfg)

        assert results_dir == cfg.results_dir
        # One call for the single period
        assert len(calls) == 1
        call = calls[0]
        assert call["instrument"] == "djia"
        assert call["workers"] == 4
        assert call["config_dir"] == config_dir
        assert call["period_results_dir"] == cfg.results_dir / "quick_test"
        assert analyzed["dir"] == cfg.results_dir
        assert analyzed["verbose"] is True


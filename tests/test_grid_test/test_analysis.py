"""Tests for core.grid_test.analysis (load_results, analyze_results_dir)."""
import pandas as pd
import pytest
from pathlib import Path

from core.grid_test.analysis import (
    load_results,
    calculate_alpha_metrics,
    analyze_results_dir,
)


def _minimal_result_row(strategy: str, outperformance: float = 10.0, win_rate: float = 55.0):
    """One row matching reporter save_results_csv schema."""
    return {
        "strategy": strategy,
        "description": "",
        "total_trades": 100,
        "winning_trades": 55,
        "losing_trades": 45,
        "no_outcome_trades": 0,
        "win_rate": win_rate,
        "average_gain": 2.0,
        "average_loss": -1.5,
        "total_gain": 110.0,
        "buy_and_hold_gain": 50.0,
        "outperformance": outperformance,
        "average_days_held": 10.0,
        "min_confidence": 0.5,
        "min_wave_size": 0.02,
        "risk_reward": 2.0,
    }


class TestLoadResults:
    def test_empty_dir_returns_empty_df(self, tmp_path):
        assert load_results(tmp_path).empty

    def test_top_level_csv_single_run(self, tmp_path):
        df_src = pd.DataFrame([_minimal_result_row("cfg_a"), _minimal_result_row("cfg_b")])
        (tmp_path / "backtest_results_20250101_120000.csv").write_text(df_src.to_csv(index=False))
        out = load_results(tmp_path)
        assert len(out) == 2
        assert list(out["period"].unique()) == ["single"]
        assert set(out["strategy"]) == {"cfg_a", "cfg_b"}

    def test_subdirs_only_multi_period(self, tmp_path):
        (tmp_path / "2018_2020").mkdir()
        (tmp_path / "2020_2022").mkdir()
        df1 = pd.DataFrame([_minimal_result_row("cfg_a", outperformance=5.0)])
        df2 = pd.DataFrame([_minimal_result_row("cfg_a", outperformance=15.0)])
        (tmp_path / "2018_2020" / "backtest_results_1.csv").write_text(df1.to_csv(index=False))
        (tmp_path / "2020_2022" / "backtest_results_2.csv").write_text(df2.to_csv(index=False))
        out = load_results(tmp_path)
        assert len(out) == 2
        assert set(out["period"]) == {"2018_2020", "2020_2022"}
        assert list(out["strategy"]) == ["cfg_a", "cfg_a"]

    def test_top_level_preferred_over_subdirs(self, tmp_path):
        (tmp_path / "sub").mkdir()
        df_top = pd.DataFrame([_minimal_result_row("top")])
        df_sub = pd.DataFrame([_minimal_result_row("sub")])
        (tmp_path / "backtest_results_top.csv").write_text(df_top.to_csv(index=False))
        (tmp_path / "sub" / "backtest_results_sub.csv").write_text(df_sub.to_csv(index=False))
        out = load_results(tmp_path)
        assert len(out) == 1
        assert out["strategy"].iloc[0] == "top"
        assert out["period"].iloc[0] == "single"


class TestCalculateAlphaMetrics:
    def test_adds_alpha_and_expectancy(self):
        df = pd.DataFrame([_minimal_result_row("x", outperformance=12.0, win_rate=60.0)])
        df["average_gain"] = 2.0
        df["average_loss"] = -1.0
        out = calculate_alpha_metrics(df)
        assert "alpha" in out.columns
        assert out["alpha"].iloc[0] == 12.0
        assert "expectancy" in out.columns


class TestAnalyzeResultsDir:
    def test_no_data_returns_false(self, tmp_path):
        assert analyze_results_dir(tmp_path, output_dir=tmp_path, verbose=False) is False

    def test_with_data_writes_reports(self, tmp_path):
        df = pd.DataFrame([_minimal_result_row("a"), _minimal_result_row("b")])
        (tmp_path / "backtest_results_1.csv").write_text(df.to_csv(index=False))
        out_dir = tmp_path / "out"
        assert analyze_results_dir(tmp_path, output_dir=out_dir, verbose=False) is True
        assert (out_dir / "analysis_report.md").exists()
        assert (out_dir / "all_results_combined.csv").exists()
        assert (out_dir / "alpha_pivot_by_config_period.csv").exists()

    def test_report_contains_expected_sections(self, tmp_path):
        df = pd.DataFrame([_minimal_result_row("cfg1")])
        (tmp_path / "backtest_results_1.csv").write_text(df.to_csv(index=False))
        analyze_results_dir(tmp_path, output_dir=tmp_path, verbose=False)
        content = (tmp_path / "analysis_report.md").read_text()
        assert "Grid Test Results Analysis" in content
        assert "Top 10 Configurations" in content
        assert "Performance by Period" in content
        assert "cfg1" in content

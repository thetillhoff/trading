from pathlib import Path

import pytest

from cli import hypothesis as cli_hypothesis


class DummyArgs:
    def __init__(self, **kwargs):
        self.category = kwargs.get("category", "all")
        self.period = kwargs.get("period", "all")
        self.instrument = kwargs.get("instrument", "djia")
        self.workers = kwargs.get("workers", 8)
        self.results_dir = kwargs.get("results_dir")


def test_cli_hypothesis_main_invokes_run_hypothesis_suite(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_parse_args():
        return DummyArgs(
            category="rsi_tests",
            period="quick_test",
            instrument="djia",
            workers=4,
            results_dir=str(tmp_path / "hypo_cli"),
        )

    monkeypatch.setattr(
        cli_hypothesis.argparse.ArgumentParser,
        "parse_args",
        staticmethod(lambda: fake_parse_args()),
    )

    def fake_run_hypothesis_suite(cfg):
        called["cfg"] = cfg
        return cfg.results_dir or Path("unused")

    # Patch the function as imported by the CLI module
    monkeypatch.setattr(cli_hypothesis, "run_hypothesis_suite", fake_run_hypothesis_suite)

    exit_code = cli_hypothesis.main()
    assert exit_code == 0

    cfg = called.get("cfg")
    assert cfg is not None
    assert cfg.category == "rsi_tests"
    assert cfg.instrument == "djia"
    assert cfg.workers == 4
    assert cfg.results_dir == Path(tmp_path / "hypo_cli")
    # period string was split into list in CLI
    assert cfg.periods == ["quick_test"]


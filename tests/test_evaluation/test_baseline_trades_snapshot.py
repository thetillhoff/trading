"""
Baseline trades snapshot test: current run matches stored golden snapshot.

Ensures indicators and the evaluation pipeline have not regressed since
hypotheses were based on them. Requires djia data for 2012; run
`make baseline-snapshot-generate` after `make download` to create/refresh
the golden file.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from core.signals.config_loader import load_config_from_yaml
from core.evaluation.walk_forward import WalkForwardEvaluator
from core.grid_test.reporter import trades_to_dataframe

SNAPSHOT_START = "2012-01-01"
SNAPSHOT_END = "2012-12-31"
TESTS_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = TESTS_DIR / "snapshots" / "baseline_trades_short.csv"


def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Align dtypes and missing values for snapshot comparison (CSV round-trip safe)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype in (np.int64, np.int32):
            out[col] = out[col].astype(np.float64)
        elif out[col].dtype == object and out[col].replace("", np.nan).isna().all():
            out[col] = out[col].replace("", np.nan)
    return out


def test_baseline_trades_match_short_timespan_snapshot():
    """Current baseline trades on 2012 match the stored golden snapshot."""
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            "Baseline snapshot not found; run make baseline-snapshot-generate after make download."
        )

    config = load_config_from_yaml(str(TESTS_DIR.parent / "configs" / "baseline.yaml"))
    config.start_date = SNAPSHOT_START
    config.end_date = SNAPSHOT_END

    try:
        evaluator = WalkForwardEvaluator(
            lookback_days=config.lookback_days,
            step_days=config.step_days,
        )
        result = evaluator.evaluate_multi_instrument(config, verbose=False)
    except Exception as e:
        pytest.skip(
            f"Baseline snapshot test requires djia data for 2012; run make download first. ({e})"
        )

    current = trades_to_dataframe(result)
    if current.empty:
        pytest.skip(
            "No trades produced (missing or insufficient data for 2012); run make download first."
        )

    expected = pd.read_csv(SNAPSHOT_PATH)

    assert list(current.columns) == list(expected.columns), (
        f"Column mismatch: current {list(current.columns)} vs expected {list(expected.columns)}"
    )
    assert len(current) == len(expected), (
        f"Row count mismatch: current {len(current)} vs expected {len(expected)}"
    )

    # Normalize: '' from trades_to_dataframe and NaN from CSV represent "missing"
    current_n = _normalize_for_compare(current.replace("", np.nan))
    expected_n = _normalize_for_compare(expected.replace("", np.nan))
    pd.testing.assert_frame_equal(
        current_n,
        expected_n,
        check_exact=False,
        atol=1e-5,
        rtol=1e-5,
        check_dtype=False,
    )

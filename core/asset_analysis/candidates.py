"""
Score instruments as trading candidates: liquidity, volatility, correlation diversity.

Normalize each metric to 0-1 and take a weighted sum for composite_score.
No per-instrument backtest score in this step.
"""
import pandas as pd
from typing import Dict, List, Optional, Union

from .analytics import compute_volatility_summary, compute_correlation_matrix
from .metadata import load_metadata


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. NaNs stay NaN; constant series -> 0.5."""
    if s.isna().all() or s.nunique() == 0:
        return s
    lo, hi = s.min(), s.max()
    if lo == hi:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def score_candidates(
    data_by_instrument: Dict[str, pd.DataFrame],
    metadata: Optional[Dict[str, Dict]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    liquidity_weight: float = 0.33,
    volatility_weight: float = 0.33,
    correlation_diversity_weight: float = 0.34,
) -> pd.DataFrame:
    """
    Rank instruments by liquidity, volatility, and correlation diversity.

    - Liquidity: higher average volume -> higher score (from metadata or OHLCV Volume).
    - Volatility: mid-range rolling vol preferred (normalized so middle is good).
    - Correlation diversity: lower average correlation to others -> higher score.

    Args:
        data_by_instrument: Map instrument name -> OHLCV DataFrame.
        metadata: Optional map name -> {average_volume, ...}. If None, load from cache.
        start_date: Analysis window start.
        end_date: Analysis window end.
        liquidity_weight: Weight for liquidity in composite (default 0.33).
        volatility_weight: Weight for volatility in composite (default 0.33).
        correlation_diversity_weight: Weight for correlation diversity (default 0.34).

    Returns:
        DataFrame with columns: instrument, name, sector, industry, liquidity_score,
        volatility_score, correlation_diversity_score, composite_score.
    """
    if not data_by_instrument:
        return pd.DataFrame(columns=[
            "instrument", "name", "sector", "industry",
            "liquidity_score", "volatility_score",
            "correlation_diversity_score", "composite_score",
        ])

    meta = metadata if metadata is not None else load_metadata()
    instruments = list(data_by_instrument.keys())

    # Liquidity: from metadata average_volume or OHLCV Volume column
    liquidity_raw = []
    for name in instruments:
        vol = None
        if meta and name in meta and meta[name].get("average_volume") is not None:
            vol = float(meta[name]["average_volume"])
        if vol is None and name in data_by_instrument:
            df = data_by_instrument[name]
            if "Volume" in df.columns and not df["Volume"].empty:
                vol = df["Volume"].mean()
        liquidity_raw.append(vol)

    liquidity_series = pd.Series(liquidity_raw, index=instruments)
    # Normalize: higher volume -> higher score (0-1)
    liquidity_score = _normalize_series(liquidity_series).reindex(instruments).fillna(0.5)

    # Volatility: from rolling vol summary; prefer mid-range (not too low, not extreme)
    vol_summary = compute_volatility_summary(
        data_by_instrument, start_date=start_date, end_date=end_date,
    )
    if vol_summary.empty or "rolling_vol" not in vol_summary.columns:
        volatility_score = pd.Series(0.5, index=instruments)
    else:
        vol_summary = vol_summary.set_index("instrument")
        vol_series = vol_summary.reindex(instruments)["rolling_vol"]
        # Prefer middle: score = 1 - 2 * |x - 0.5| after normalizing vol to 0-1
        vol_norm = _normalize_series(vol_series)
        volatility_score = (1.0 - (2 * (vol_norm.fillna(0.5) - 0.5).abs())).clip(0, 1)
        volatility_score = volatility_score.reindex(instruments).fillna(0.5)

    # Correlation diversity: lower mean absolute correlation to others -> higher score
    corr_matrix = compute_correlation_matrix(
        data_by_instrument, start_date=start_date, end_date=end_date,
    )
    if corr_matrix.empty or corr_matrix.shape[1] < 2:
        correlation_diversity_score = pd.Series(0.5, index=instruments)
    else:
        # Mean absolute correlation (excluding diagonal)
        mean_corr = {}
        for name in instruments:
            if name not in corr_matrix.columns:
                mean_corr[name] = 0.5
                continue
            col = corr_matrix[name].drop(index=[name], errors="ignore")
            mean_corr[name] = col.abs().mean() if not col.empty else 0.5
        div_series = pd.Series(mean_corr)
        # Lower correlation -> better diversity -> higher score; so score = 1 - normalized
        div_norm = _normalize_series(div_series)
        correlation_diversity_score = (1.0 - div_norm).reindex(instruments).fillna(0.5)

    # Composite
    composite = (
        liquidity_weight * liquidity_score.fillna(0.5)
        + volatility_weight * volatility_score.fillna(0.5)
        + correlation_diversity_weight * correlation_diversity_score.fillna(0.5)
    )

    # Name, sector, industry from metadata (when available)
    full_name = []
    for tick in instruments:
        m = meta.get(tick, {}) if meta else {}
        full_name.append(m.get("long_name") or m.get("short_name"))
    sector = [meta.get(tick, {}).get("sector") if meta else None for tick in instruments]
    industry = [meta.get(tick, {}).get("industry") if meta else None for tick in instruments]

    out = pd.DataFrame({
        "instrument": instruments,
        "name": full_name,
        "sector": sector,
        "industry": industry,
        "liquidity_score": liquidity_score.reindex(instruments).values,
        "volatility_score": volatility_score.reindex(instruments).values,
        "correlation_diversity_score": correlation_diversity_score.reindex(instruments).values,
        "composite_score": composite.reindex(instruments).values,
    })
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)

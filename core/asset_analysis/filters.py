"""
Filter candidate ranking by ASSET_CATEGORIES.md (Excluded Categories).

Drops rows where sector or industry contains any excluded keyword (case-insensitive).
"""
import pandas as pd
from typing import List

# From ASSET_CATEGORIES.md "Excluded Categories (Do Not Trade)".
# Keywords match both hand-written wording and yfinance sector/industry values.
EXCLUDED_KEYWORDS: List[str] = [
    # Oil & Gas
    "oil & gas",
    "oil and gas",
    "oil & gas e&p",
    "oil & gas equipment",
    "oil & gas integrated",
    "oil & gas midstream",
    "oil & gas refining",
    # Coal
    "coal",
    # Plastics & Chemicals
    "plastics",
    "chemicals",
    "specialty chemicals",
    # Nuclear Energy
    "nuclear",
    # Livestock
    "livestock",
    "farm products",
    # Food Production & Processing
    "food production",
    "food processing",
    "packaged foods",
    "food distribution",
    "confectioners",
    # Cryptocurrencies
    "cryptocurrency",
    "crypto",
    # Real Estate & REITs
    "real estate",
    "reit",
    "reit -",
    "residential construction",
    "real estate services",
]


def filter_candidates_by_asset_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where sector or industry matches ASSET_CATEGORIES excluded list.

    Uses substring match (case-insensitive) on sector and industry columns.
    Rows with missing sector/industry are kept.

    Args:
        df: Candidate ranking DataFrame with columns instrument, name, sector, industry, ...

    Returns:
        Filtered DataFrame (copy) with excluded categories removed.
    """
    if df.empty or ("sector" not in df.columns and "industry" not in df.columns):
        return df.copy()

    def _matches_excluded(val) -> bool:
        if pd.isna(val) or not isinstance(val, str):
            return False
        v = val.lower()
        return any(k in v for k in EXCLUDED_KEYWORDS)

    sector_match = df["sector"].map(_matches_excluded) if "sector" in df.columns else pd.Series(False, index=df.index)
    industry_match = df["industry"].map(_matches_excluded) if "industry" in df.columns else pd.Series(False, index=df.index)
    drop = sector_match | industry_match
    return df.loc[~drop].copy()

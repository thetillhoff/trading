"""
Discover available assets (e.g. S&P 500, NASDAQ-100, DAX, DJIA from Wikipedia). Cache-first.
"""
import io
import urllib.request
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..data.download import DATA_DIR

# Wikipedia returns 403 for default User-Agent; use a browser-like one
USER_AGENT = "Mozilla/5.0 (compatible; trading-asset-discovery/1.0; +https://github.com)"

# source -> (url, cache_filename, symbol_column, table_index or None, dot_to_hyphen for Yahoo)
SOURCES = {
    "sp500": (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "available_assets_sp500.csv",
        "Symbol",
        0,
        True,
    ),
    "nasdaq100": (
        "https://en.wikipedia.org/wiki/NASDAQ-100",
        "available_assets_nasdaq100.csv",
        "Ticker",
        None,
        True,
    ),
    "dax": (
        "https://en.wikipedia.org/wiki/DAX",
        "available_assets_dax.csv",
        "Ticker",
        None,
        False,
    ),
    "djia": (
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "available_assets_djia.csv",
        "Symbol",
        None,
        True,
    ),
}


def _cache_path(source: str) -> Path:
    _, filename, _, _, _ = SOURCES[source]
    return DATA_DIR / filename


def _find_components_table(
    tables: list,
    symbol_column: str,
    min_rows: int = 10,
) -> Optional[pd.DataFrame]:
    """Return first table that has symbol_column and at least min_rows."""
    for df in tables:
        if symbol_column in df.columns and len(df) >= min_rows:
            return df
    return None


def get_available_assets(
    source: str = "sp500",
    refresh: bool = False,
) -> List[str]:
    """
    Get list of available ticker symbols. Cache-first.

    - source="sp500": S&P 500 constituents (Wikipedia). Cached to available_assets_sp500.csv.
    - source="nasdaq100": NASDAQ-100 constituents. Cached to available_assets_nasdaq100.csv.
    - source="dax": DAX 40 constituents (XETR, e.g. ADS.DE). Cached to available_assets_dax.csv.
    - source="djia": Dow Jones Industrial Average 30 constituents. Cached to available_assets_djia.csv.
    - On rerun loads from file unless refresh=True.
    - Returns list of ticker symbols.
    """
    if source not in SOURCES:
        return []

    path = _cache_path(source)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        if "Symbol" in df.columns:
            return df["Symbol"].astype(str).str.strip().tolist()
        if len(df.columns) >= 1:
            return df.iloc[:, 0].astype(str).str.strip().tolist()
        return []

    url, _, symbol_column, table_index, dot_to_hyphen = SOURCES[source]

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode()
        tables = pd.read_html(io.StringIO(html))
        if not tables:
            raise ValueError("Wikipedia page returned no tables")

        if table_index is not None:
            df = tables[table_index]
        else:
            df = _find_components_table(tables, symbol_column)
            if df is None:
                raise ValueError(f"No table with column '{symbol_column}' and enough rows")

        if symbol_column not in df.columns and len(df.columns) >= 1:
            symbols = df.iloc[:, 0].astype(str).str.strip()
        else:
            symbols = df[symbol_column].astype(str).str.strip()
        if dot_to_hyphen:
            symbols = symbols.str.replace(".", "-", regex=False)
        tickers = symbols.tolist()
        if not tickers:
            raise ValueError("No tickers extracted from table")
    except Exception as e:
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        raise RuntimeError(
            f"Failed to discover {source} tickers from Wikipedia: {type(e).__name__}: {msg}. "
            "Ensure lxml is installed and network is available."
        ) from e

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({"Symbol": tickers})
    name_col = "Company" if "Company" in df.columns else ("Security" if "Security" in df.columns else None)
    if name_col:
        df_out["Name"] = df[name_col].values
    df_out.to_csv(path, index=False)
    return tickers


def get_all_sources() -> List[str]:
    """Return list of supported discovery source keys."""
    return list(SOURCES.keys())


def get_available_assets_all_sources(refresh: bool = False) -> List[str]:
    """
    Get union of ticker symbols from all discovery sources. Cache-per-source;
    each source uses its own CSV; this returns deduplicated combined list
    (exact string match only; BRK.B and BRK-B from different sources both kept).
    """
    seen: set = set()
    result: List[str] = []
    for source in get_all_sources():
        for t in get_available_assets(source=source, refresh=refresh):
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result

"""
Fetch and persist instrument metadata from yfinance (cache-first).

Metadata is stored in data/instrument_metadata.json. On rerun we load from file
unless refresh is requested; we only call yfinance when the file is missing or
--refresh-metadata is passed.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# JSON metadata lives in data/ (DATA_DIR). Ticker OHLCV lives in data/tickers/ (TICKERS_DIR).
from ..data.download import DATA_DIR, INSTRUMENTS

METADATA_FILENAME = "instrument_metadata.json"
AVAILABLE_ASSETS_METADATA_FILENAME = "available_assets_metadata.json"


def _metadata_path() -> Path:
    """Instrument metadata JSON: data/instrument_metadata.json."""
    return DATA_DIR / METADATA_FILENAME


def available_assets_metadata_path() -> Path:
    """Discovered-assets metadata JSON: data/available_assets_metadata.json (not in data/tickers/)."""
    return DATA_DIR / AVAILABLE_ASSETS_METADATA_FILENAME


def _sanitize_str(v: Any, max_len: int = 500) -> Optional[str]:
    """Return a clean string for storage; None if value looks like HTML or is too long."""
    if v is None or not isinstance(v, str):
        return None
    s = v.strip()
    if not s or len(s) > max_len:
        return None
    low = s.lower()
    if low.startswith("<!doctype") or low.startswith("<html") or "<body" in low:
        return None
    return s


def _normalize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small, JSON-serializable subset of yfinance Ticker.info."""
    return {
        "sector": _sanitize_str(info.get("sector")),
        "industry": _sanitize_str(info.get("industry")),
        "market_cap": info.get("marketCap"),
        "average_volume": info.get("averageVolume"),
        "short_name": _sanitize_str(info.get("shortName")),
        "long_name": _sanitize_str(info.get("longName")),
    }


def fetch_metadata(
    instruments: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch metadata for instruments or raw tickers. Cache-first: load from file
    if it exists and refresh is False; only call yfinance when missing or refresh=True.

    Args:
        instruments: Instrument names (keys in INSTRUMENTS). Used only if tickers is None.
        tickers: Raw ticker symbols (e.g. from discovery). Saved to available_assets_metadata.json.
        refresh: If True, fetch from yfinance and overwrite cache.

    Returns:
        Dict mapping instrument name or ticker -> normalized metadata dict.
    """
    import yfinance as yf

    if tickers is not None:
        path = available_assets_metadata_path()
        names_or_tickers = list(tickers)
        if path.exists() and not refresh:
            cached = load_metadata(path=path)
            # Only use cache if it covers all requested tickers (e.g. all sources vs sp500-only)
            if cached and set(names_or_tickers).issubset(cached.keys()):
                return {k: cached[k] for k in names_or_tickers}
        names_or_tickers = list(tickers)
        result = _fetch_for_symbols(yf, names_or_tickers, by_ticker=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_metadata(result, path=path)
        return result

    path = _metadata_path()
    if path.exists() and not refresh:
        return load_metadata()
    names = list(instruments) if instruments is not None else list(INSTRUMENTS.keys())
    result = _fetch_for_symbols(yf, names, by_ticker=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_metadata(result, path=path)
    return result


def _fetch_for_symbols(
    yf: Any,
    names_or_tickers: List[str],
    by_ticker: bool,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for key in names_or_tickers:
        if by_ticker:
            ticker_symbol = key
        else:
            if key not in INSTRUMENTS:
                continue
            ticker_symbol, _ = INSTRUMENTS[key]
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info or {}
            result[key] = _normalize_info(info)
        except Exception:
            result[key] = _normalize_info({})
    return result


def load_metadata(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from JSON file. Returns empty dict if file does not exist.

    Args:
        path: Path to JSON file. If None, use default data/instrument_metadata.json.

    Returns:
        Dict mapping instrument name -> normalized metadata dict.
    """
    p = path if path is not None else _metadata_path()
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata: Dict[str, Dict[str, Any]], path: Optional[Path] = None) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Dict mapping instrument name -> normalized metadata dict.
        path: Path to JSON file. If None, use default data/instrument_metadata.json.
    """
    p = path if path is not None else _metadata_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

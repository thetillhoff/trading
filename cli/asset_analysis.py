#!/usr/bin/env python3
"""
Asset analysis CLI: metadata, returns/vol/correlation, candidate scoring.

Uses cached OHLCV and cached metadata; only fetches when missing or --refresh-metadata.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root on path (Docker: PYTHONPATH=/app)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.download import download_instrument, download_ticker, INSTRUMENTS, TICKERS_DIR
from core.data.loader import DataLoader
from core.asset_analysis.discovery import get_available_assets, get_available_assets_all_sources
from core.asset_analysis.metadata import (
    fetch_metadata,
    load_metadata,
    available_assets_metadata_path,
)
from core.asset_analysis.analytics import (
    compute_volatility_summary,
    compute_correlation_matrix,
)
from core.asset_analysis.candidates import score_candidates
from core.asset_analysis.filters import filter_candidates_by_asset_categories


def _ensure_data(
    instruments: list,
    start_date: str,
    end_date: str,
) -> dict:
    """Ensure OHLCV cache exists (cache-first); load and return Dict[name, DataFrame]."""
    data_by_instrument = {}
    for name in instruments:
        if name not in INSTRUMENTS:
            continue
        download_instrument(name, force_refresh=False, start_date=start_date)
        try:
            loaded = DataLoader.from_instrument(
                name, start_date=start_date, end_date=end_date, column=None
            )
        except (FileNotFoundError, ValueError, AttributeError):
            continue
        if isinstance(loaded, pd.DataFrame):
            data_by_instrument[name] = loaded
        else:
            data_by_instrument[name] = loaded.to_frame()
    return data_by_instrument


def _ensure_data_tickers(
    tickers: list,
    start_date: str,
    end_date: str,
    update_stale: bool = False,
) -> tuple[dict, list]:
    """Download OHLCV for discovered tickers (cache-first to data/tickers/); load and return (Dict[ticker, DataFrame], skipped_tickers)."""
    # Resolve path so we see where cache is (e.g. in Docker /app/data/tickers)
    cache_dir = TICKERS_DIR.resolve()
    have_cache = sum(1 for t in tickers if (TICKERS_DIR / f"{t}.csv").exists())
    n = len(tickers)
    print(f"  OHLCV cache: {cache_dir} ({have_cache}/{n} files present)", flush=True)
    data_by_instrument = {}
    skipped = []
    from_cache = 0
    downloaded = 0
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  OHLCV: {i + 1}/{n} {ticker}...", flush=True)
        _, used_cache = download_ticker(
            ticker,
            force_refresh=False,
            start_date=start_date,
            quiet=True,
            update_stale=update_stale,
        )
        if used_cache:
            from_cache += 1
        else:
            downloaded += 1
        try:
            loaded = DataLoader.from_ticker(
                ticker, start_date=start_date, end_date=end_date, column=None
            )
        except (FileNotFoundError, ValueError, AttributeError):
            skipped.append(ticker)
            continue
        if isinstance(loaded, pd.DataFrame):
            data_by_instrument[ticker] = loaded
        else:
            data_by_instrument[ticker] = loaded.to_frame()
    if downloaded:
        print(f"  OHLCV: used cache for {from_cache}, network for {downloaded} (no cache or refresh).", flush=True)
    return data_by_instrument, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Asset analysis: metadata, returns/vol/correlation, candidate scoring (cache-first)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fetch-metadata",
        action="store_true",
        help="Fetch and save instrument metadata (only hits network when file missing or --refresh-metadata)",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Force re-fetch metadata from yfinance; otherwise use cached file",
    )
    parser.add_argument(
        "--all-assets",
        action="store_true",
        help="Use discovered assets for metadata; ignores --instruments for metadata",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["sp500", "nasdaq100", "dax", "djia"],
        help="Limit to this discovery source when using --all-assets; if omitted, all sources are used",
    )
    parser.add_argument(
        "--refresh-assets",
        action="store_true",
        help="Force re-fetch asset list from Wikipedia; only with --all-assets",
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        help="Instrument names (default: all from INSTRUMENTS)",
    )
    parser.add_argument(
        "--start-date", "-s",
        default="2000-01-01",
        help="Analysis window start (default: 2000-01-01)",
    )
    parser.add_argument(
        "--end-date", "-e",
        default="2024-12-31",
        help="Analysis window end (default: 2024-12-31)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run returns/vol/correlation and candidate scoring (uses cached OHLCV and metadata)",
    )
    parser.add_argument(
        "--update-tickers",
        action="store_true",
        help="When using --all-assets --analyze: try to update stale ticker OHLCV (one request per ticker when cache is old); default is use cache as-is",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for report (default: data/asset_analysis)",
    )
    DEFAULT_CANDIDATES_CSV = "data/asset_analysis/candidate_ranking.csv"
    parser.add_argument(
        "--download-candidates",
        type=str,
        default=None,
        nargs="?",
        const=DEFAULT_CANDIDATES_CSV,
        metavar="CSV",
        help=f"Download OHLCV for tickers from this candidate ranking CSV (default: {DEFAULT_CANDIDATES_CSV}); use --top N to limit rows",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Output only top N: with --download-candidates = rows from CSV to download; with --analyze = rows to write to candidate_ranking CSVs (default: all). Metadata and OHLCV are always for all tickers (cache-first).",
    )
    args = parser.parse_args()

    use_all_assets = args.all_assets
    instruments = args.instruments if args.instruments else list(INSTRUMENTS.keys())
    out_dir = Path(args.output) if args.output else Path("data") / "asset_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download OHLCV for tickers from a candidate ranking CSV
    if args.download_candidates is not None:
        csv_path = Path(args.download_candidates)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        if not csv_path.exists():
            print(f"Error: candidate CSV not found: {csv_path}", flush=True)
            return 1
        df = pd.read_csv(csv_path)
        if "instrument" not in df.columns:
            print("Error: CSV must have an 'instrument' column.", flush=True)
            return 1
        tickers_to_download = df.head(args.top)["instrument"].tolist() if args.top is not None else df["instrument"].tolist()
        print(f"Downloading OHLCV for {len(tickers_to_download)} candidates from {csv_path.name}...", flush=True)
        for i, ticker in enumerate(tickers_to_download, 1):
            print(f"  {i}/{len(tickers_to_download)} {ticker}...", flush=True)
            download_ticker(
                ticker,
                force_refresh=False,
                start_date=args.start_date,
                quiet=False,
                update_stale=args.update_tickers,
            )
        print(f"Done. Data in data/tickers/", flush=True)
        return 0

    print("Asset analysis", flush=True)
    if use_all_assets:
        if args.source is None:
            tickers = get_available_assets_all_sources(refresh=args.refresh_assets)
            print(f"  Assets (all sources): {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})", flush=True)
        else:
            tickers = get_available_assets(source=args.source, refresh=args.refresh_assets)
            print(f"  Assets ({args.source}): {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})", flush=True)
    else:
        print(f"  Instruments: {len(instruments)} ({', '.join(instruments[:5])}{'...' if len(instruments) > 5 else ''})", flush=True)

    # Metadata: fetch or load from cache
    if args.fetch_metadata or args.refresh_metadata:
        if args.refresh_metadata:
            print("  Force re-fetching metadata from yfinance...", flush=True)
        else:
            print("  Ensuring metadata (load from cache or fetch if missing)...", flush=True)
        if use_all_assets:
            meta = fetch_metadata(tickers=tickers, refresh=args.refresh_metadata)
            print(f"  Metadata ready: {len(meta)} tickers → data/available_assets_metadata.json", flush=True)
        else:
            meta = fetch_metadata(instruments=instruments, refresh=args.refresh_metadata)
            print(f"  Metadata ready: {len(meta)} instruments → data/instrument_metadata.json", flush=True)
    else:
        if use_all_assets:
            meta = load_metadata(path=available_assets_metadata_path())
            if meta:
                print(f"  Metadata loaded from cache ({len(meta)} tickers).", flush=True)
            else:
                print("  No cached metadata for discovered assets. Use --fetch-metadata to fetch.", flush=True)
        else:
            meta = load_metadata()
            if meta:
                print(f"  Metadata loaded from cache ({len(meta)} instruments).", flush=True)
            else:
                print("  No cached metadata. Use --fetch-metadata to fetch.", flush=True)

    if not args.analyze and not args.fetch_metadata:
        print("Done. Use --fetch-metadata to fetch metadata, or --analyze to run full analysis.", flush=True)
        return 0

    if args.fetch_metadata and not args.analyze:
        print("Done.", flush=True)
        return 0

    if args.analyze:
        if use_all_assets:
            # Always load metadata and OHLCV for all tickers (cache-first per ticker)
            if args.update_tickers:
                print("  Downloading/loading OHLCV (cache-first, --update-tickers to refresh stale)...", flush=True)
            else:
                print("  Loading OHLCV from cache (use --update-tickers to refresh stale data)...", flush=True)
            data_by_instrument, skipped_tickers = _ensure_data_tickers(
                tickers,
                args.start_date,
                args.end_date,
                update_stale=args.update_tickers,
            )
            if skipped_tickers:
                print(f"  Skipped {len(skipped_tickers)} ticker(s): {', '.join(skipped_tickers)}", flush=True)
        else:
            print("  Loading OHLCV (cache-first)...", flush=True)
            data_by_instrument = _ensure_data(instruments, args.start_date, args.end_date)
        if not data_by_instrument:
            msg = "No data available. "
            if use_all_assets:
                msg += "Run with --all-assets --analyze (tickers download to data/tickers/)."
            else:
                msg += "Run make download first for the requested instruments."
            print(msg, flush=True)
            return 1
        print(f"  Loaded {len(data_by_instrument)} instruments.", flush=True)

        # Volatility summary
        print("  Computing volatility summary...", flush=True)
        vol_summary = compute_volatility_summary(
            data_by_instrument,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        vol_path = out_dir / "volatility_summary.csv"
        vol_summary.to_csv(vol_path, index=False)
        print(f"  Wrote {vol_path}", flush=True)

        # Correlation matrix
        print("  Computing correlation matrix...", flush=True)
        corr_matrix = compute_correlation_matrix(
            data_by_instrument,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        corr_path = out_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_path)
        print(f"  Wrote {corr_path}", flush=True)

        # Candidate ranking (full; --top only trims what we write)
        print("  Scoring candidates (liquidity, volatility, correlation diversity)...", flush=True)
        full_candidates = score_candidates(
            data_by_instrument,
            metadata=meta,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        candidates_df = full_candidates.head(args.top) if args.top is not None else full_candidates
        candidates_path = out_dir / "candidate_ranking.csv"
        candidates_df.to_csv(candidates_path, index=False)
        print(f"  Wrote {candidates_path}", flush=True)

        # Filtered copy per ASSET_CATEGORIES.md (excluded categories removed), then --top
        filtered_full = filter_candidates_by_asset_categories(full_candidates)
        filtered_df = filtered_full.head(args.top) if args.top is not None else filtered_full
        filtered_path = out_dir / "candidate_ranking_filtered.csv"
        filtered_df.to_csv(filtered_path, index=False)
        print(f"  Wrote {filtered_path} ({len(filtered_df)} rows)", flush=True)

        # Short summary to stdout
        n_show = min(10, len(candidates_df))
        print(f"\nTop {n_show} candidates (composite_score):", flush=True)
        print(candidates_df.head(n_show).to_string(index=False), flush=True)
        print(f"\nDone. Results in {out_dir}/", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

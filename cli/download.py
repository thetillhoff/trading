#!/usr/bin/env python3
"""
Data download CLI.

Downloads historical data for any instrument.
"""
import argparse
import sys
from pathlib import Path

# Add core to path
core_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_dir.parent))

from core.data.scraper import download_instrument, download_all, list_instruments, get_available_instruments


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for trading instruments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available instruments
    python -m cli.download --list

    # Download specific instrument
    python -m cli.download djia

    # Download multiple instruments
    python -m cli.download djia sp500 gold

    # Download all instruments
    python -m cli.download

    # Force refresh (re-download)
    python -m cli.download --refresh djia
        """
    )
    
    parser.add_argument(
        "instruments",
        nargs="*",
        help="Instrument names to download (default: all)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available instruments",
    )
    parser.add_argument(
        "--refresh", "-r",
        action="store_true",
        help="Force refresh (re-download even if cached)",
    )
    parser.add_argument(
        "--start-date", "-s",
        default="1990-01-01",
        help="Start date for historical data (default: 1990-01-01)",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_instruments()
        return 0
    
    if args.instruments:
        # Download specific instruments
        for name in args.instruments:
            result = download_instrument(
                name,
                force_refresh=args.refresh,
                start_date=args.start_date
            )
            if result is None:
                print(f"Failed to download {name}")
                return 1
    else:
        # Download all instruments
        results = download_all(
            force_refresh=args.refresh,
            start_date=args.start_date
        )
        if not results:
            print("Failed to download any instruments")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

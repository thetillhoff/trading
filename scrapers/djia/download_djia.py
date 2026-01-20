#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "djia_data.csv")

def main():
    # Check if CSV file exists
    if os.path.exists(CSV_FILE):
        print(f"Loading data from {CSV_FILE}...")
        djia = pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
        print("Data loaded from CSV file.")
    else:
        print("Downloading data from yfinance...")
        djia = yf.download("^DJI", start="1900-01-01")
        print(f"Saving data to {CSV_FILE}...")
        djia.to_csv(CSV_FILE)
        print("Data saved to CSV file.")
    
    print("\nFirst few rows:")
    print(djia.head())
    print(f"\nData shape: {djia.shape}")
    print(f"Date range: {djia.index.min()} to {djia.index.max()}")

if __name__ == "__main__":
    main()

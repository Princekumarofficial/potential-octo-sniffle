"""Process raw US 1-minute stock CSV files into feature-engineered Parquet files.

Regular trading hours: 09:30 to 15:59 EST (390 bars per full session).
Outputs to datasets/processed_datasets/us_stocks/{TICKER}.parquet.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Ensure full_predictor is importable
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from helpers.feature_engineering import fill_short_gaps, add_features

RAW = ["open", "high", "low", "close", "volume"]
DEFAULT_INPUT_DIR = BASE_DIR / "datasets" / "us_stocks"
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "processed_datasets" / "us_stocks"


def extract_symbol(path: Path) -> str:
    stem = path.stem
    if "_clean_1min" in stem:
        return stem.replace("_clean_1min", "")
    if "_1min" in stem:
        return stem.replace("_1min", "")
    return stem.split("_")[0]


def process_one_stock(task):
    csv_path, output_dir = task
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    ticker = extract_symbol(csv_path)
    target_parquet = output_dir / f"{ticker}.parquet"

    t0 = time.perf_counter()
    try:
        # 1. Read raw CSV
        df = pd.read_csv(csv_path)
        date_col = "datetime" if "datetime" in df.columns else ("date" if "date" in df.columns else df.columns[0])
        df["date"] = pd.to_datetime(df[date_col])

        # 2. Normalize columns to lowercase OHLCV
        rename_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename_map).set_index("date").sort_index()

        # Handle potential duplicates
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        df = df[RAW]
        for col in RAW:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Construct US regular market hours grid (09:30 to 15:59)
        # Filter raw to regular trading hours if any pre/post market exists
        minutes = df.index.hour * 60 + df.index.minute
        regular_mask = (minutes >= 570) & (minutes <= 959)  # 09:30 to 15:59
        df = df[regular_mask]

        days = pd.Series(df.index.normalize().unique())
        all_minutes = [
            pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=15, minutes=59), freq="1min")
            for d in days
        ]
        if not all_minutes:
            return ticker, False, f"No trading days found in {csv_path.name}", 0.0

        expected_grid = pd.DatetimeIndex(np.concatenate(all_minutes))

        # 4. Reindex and audit missing bars
        aligned = df.reindex(expected_grid)
        aligned.index.name = "date"
        aligned["is_missing_bar"] = ~expected_grid.isin(df.index)

        prices = aligned[RAW[:4]]
        aligned["is_valid_bar"] = (
            np.isfinite(aligned[RAW]).all(axis=1)
            & prices.gt(0).all(axis=1)
            & aligned.volume.ge(0)
            & aligned.high.ge(prices.max(axis=1))
            & aligned.low.le(prices.min(axis=1))
        )

        # 5. Gap fill and indicator computation
        fill_short_gaps(aligned)
        transformed, features, _ = add_features(aligned, include_garch=False)

        # 6. Save as Parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        transformed.to_parquet(target_parquet, engine="pyarrow", compression="zstd")

        elapsed = time.perf_counter() - t0
        return ticker, True, f"Saved {len(transformed):,} bars ({len(features)} features) in {elapsed:.1f}s", elapsed

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return ticker, False, str(exc), elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory with raw US stock CSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for processed parquet files.")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel worker processes.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of stocks to process (0 = all).")
    parser.add_argument("--stocks", nargs="+",
                        help="Specific stock tickers to process.")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {args.input_dir}")
        return

    if args.stocks:
        targets = set(s.upper() for s in args.stocks)
        files = [p for p in files if extract_symbol(p).upper() in targets]
    elif args.limit > 0:
        files = files[:args.limit]

    print(f"Processing {len(files)} US stocks with {args.workers} workers...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(f, args.output_dir) for f in files]
    success_count = 0
    t_start = time.perf_counter()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, (ticker, ok, msg, duration) in enumerate(executor.map(process_one_stock, tasks), 1):
                status = "OK" if ok else "FAILED"
                print(f"[{idx}/{len(files)}] {ticker:6s} -> {status}: {msg}", flush=True)
                if ok:
                    success_count += 1
    else:
        for idx, task in enumerate(tasks, 1):
            ticker, ok, msg, duration = process_one_stock(task)
            status = "OK" if ok else "FAILED"
            print(f"[{idx}/{len(files)}] {ticker:6s} -> {status}: {msg}", flush=True)
            if ok:
                success_count += 1

    total_time = time.perf_counter() - t_start
    print(f"\nCompleted: {success_count}/{len(files)} stocks processed successfully in {total_time:.1f}s.")


if __name__ == "__main__":
    main()

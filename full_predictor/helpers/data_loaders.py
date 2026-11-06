"""Read processed stock Parquet files without changing their timestamps or NaNs."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "processed_datasets" / "NIFTY50"


def list_processed_stocks() -> list[str]:
    """Return available stock symbols in alphabetical order."""
    return sorted(path.stem for path in PROCESSED_DATA_DIR.glob("*.parquet") if path.is_file())


def load_processed_stock(
    stock: str, *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load one stock by symbol, optionally reading only the requested columns.

    The saved datetime index, missing values, and requested metadata columns
    are preserved. For example, use columns=["close", "segment_id"] to read
    only the closing price and segment identifier.
    """
    symbol = stock.strip().upper()
    if not symbol or any(char in symbol for char in '/\\:') or symbol in {'.', '..'}:
        raise ValueError(f"Expected a stock symbol, got {stock!r}")
    path = PROCESSED_DATA_DIR / f"{symbol}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No processed Parquet file for {symbol}: {path}")
    return pd.read_parquet(path, engine="pyarrow", columns=None if columns is None else list(columns))

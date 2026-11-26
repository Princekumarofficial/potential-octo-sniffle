"""Read processed Parquet files and build gap-aware forecasting datasets."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "datasets/processed_datasets" / "NIFTY50"


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


class StockWindowDataset(Dataset):
    """Return (lookback × features, scalar forward return) tensors for one stock.

    A horizon of one predicts the simple return from the final input close to
    the next close. Longer horizons predict the cumulative simple return.
    Stride is measured from the first candidate row, before filtering.
    """

    def __init__(self, frame: pd.DataFrame, feature_columns: Sequence[str], *,
                 lookback: int = 30, horizon: int = 1, stride: int = 1):
        from .feature_engineering import RAW, add_features

        if any(type(value) is not int or value < 1 for value in (lookback, horizon, stride)):
            raise ValueError("lookback, horizon, and stride must be positive integers")
        if isinstance(feature_columns, str):
            raise TypeError("feature_columns must be a sequence of column names")
        self.feature_columns = tuple(feature_columns)
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must contain distinct column names")
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
                or not frame.index.is_monotonic_increasing or not frame.index.is_unique):
            raise ValueError("Expected a sorted, unique DatetimeIndex without NaT")

        self.lookback = lookback
        observed = frame.loc[:, [*RAW, "is_missing_bar", "is_valid_bar"]].copy()
        unavailable = observed.is_missing_bar | ~observed.is_valid_bar
        if "is_imputed_bar" in frame:
            unavailable |= frame.is_imputed_bar
        observed.loc[unavailable, RAW] = np.nan
        observed["is_valid_bar"] = ~unavailable

        # Recompute from observed bars so later indicators cannot inherit fills.
        features, available_columns, _ = add_features(
            observed, include_garch="garch_daily_volatility" in self.feature_columns
        )
        unknown = set(self.feature_columns) - set([*RAW, *available_columns])
        if unknown:
            raise ValueError(f"Unknown input features: {sorted(unknown)}")
        values = features.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float32, copy=True)
        self._values = torch.from_numpy(values)

        starts = np.arange(0, max(0, len(features) - lookback - horizon + 1), stride)
        input_ends = starts + lookback - 1
        target_ends = input_ends + horizon
        segments = features.segment_id.to_numpy()
        bad_feature_rows = ~np.isfinite(values).all(axis=1)
        bad_prefix = np.r_[0, np.cumsum(bad_feature_rows)]
        complete_inputs = bad_prefix[input_ends + 1] == bad_prefix[starts]
        continuous = segments[starts] == segments[target_ends]
        valid_bars = observed.is_valid_bar.to_numpy()
        eligible = complete_inputs & continuous & valid_bars[starts] & valid_bars[target_ends]

        close = observed.close.to_numpy(dtype=float)
        eligible_starts = starts[eligible]
        targets = (close[target_ends[eligible]] / close[input_ends[eligible]] - 1).astype(np.float32)
        finite_targets = np.isfinite(targets)
        self._starts = eligible_starts[finite_targets]
        self._targets = torch.from_numpy(targets[finite_targets])
        self.target_timestamps = frame.index.take(self._starts + lookback - 1 + horizon)

    @classmethod
    def from_stock(cls, stock: str, feature_columns: Sequence[str], **window_options):
        """Build a dataset from a stock's processed Parquet file."""
        return cls(load_processed_stock(stock), feature_columns, **window_options)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self._starts[index])
        return self._values[start:start + self.lookback], self._targets[index]


class MultiStockWindowDataset(ConcatDataset):
    """Combine stock datasets that share the same feature order and window settings."""

    def __init__(self, stocks: Sequence[str], feature_columns: Sequence[str], *,
                 lookback: int = 30, horizon: int = 1, stride: int = 1):
        if isinstance(stocks, str) or not stocks:
            raise ValueError("Provide a nonempty sequence of stock symbols")
        self.stocks = tuple(stock.strip().upper() for stock in stocks)
        if len(set(self.stocks)) != len(self.stocks):
            raise ValueError("Stock symbols must be distinct")
        super().__init__([
            StockWindowDataset.from_stock(stock, feature_columns, lookback=lookback,
                                          horizon=horizon, stride=stride)
            for stock in self.stocks
        ])

    @property
    def samples_per_stock(self) -> dict[str, int]:
        """Number of eligible windows for each requested stock."""
        return {stock: len(dataset) for stock, dataset in zip(self.stocks, self.datasets)}

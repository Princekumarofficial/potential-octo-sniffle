# Data loaders and forecasting datasets

[helpers/data_loaders.py](../helpers/data_loaders.py) reads the saved Parquet outputs and provides PyTorch datasets for single-stock and multi-stock forecasts. Run the examples from inside `full_predictor` with NumPy, pandas, PyTorch, pyarrow, and arch installed.

| API | Contract |
| --- | --- |
| `list_processed_stocks()` | Returns sorted symbols found in `processed_datasets/NIFTY50/`. |
| `load_processed_stock(stock, columns=None)` | Reads one symbol, optionally projecting columns. Preserves the saved datetime index and NaNs. |
| `StockWindowDataset(frame, feature_columns, *, lookback=30, horizon=1, stride=1)` | Indexes eligible windows in one stock frame. |
| `StockWindowDataset.from_stock(stock, feature_columns, **window_options)` | Loads the stock's Parquet and builds its window dataset. |
| `MultiStockWindowDataset(stocks, feature_columns, *, lookback=30, horizon=1, stride=1)` | Concatenates individual datasets using the same feature order and window settings. `samples_per_stock` reports eligible counts. |

## Sample definition

- **Inputs.** A sorted, unique datetime index and OHLCV plus `is_missing_bar` and `is_valid_bar`. The optional `is_imputed_bar` flag is also respected.
- **Features.** The dataset masks unavailable bars and recomputes indicators from observed OHLCV. This removes the influence of two-sided fills on later indicators. Specify only the input columns you want. Missing values in other features do not affect eligibility.
- **Window.** The lookback and target must stay in one continuous segment for one stock. Selected input values and the target must be finite. Stride applies to candidate starts before filtering.
- **Target.** For last input row `t` and horizon `h`, the scalar is `close[t+h] / close[t] - 1`, in decimal units. The target close comes from an observed, valid bar.
- **Output.** Each item is `(X, y)` with float32 shapes `(lookback, feature_count)` and `()`. PyTorch batches these as `(batch, lookback, feature_count)` and `(batch,)`.

## Example

```python
from torch.utils.data import DataLoader
from helpers.data_loaders import MultiStockWindowDataset

dataset = MultiStockWindowDataset(
    ["HYUNDAI", "RELIANCE"],
    ["returns", "rsi_14", "log_volume"],
    lookback=30,
)
print(dataset.samples_per_stock)
if len(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    X, y = next(iter(loader))
```

For training, construct datasets from frames restricted to each chronological split and use common cutoffs across stocks. Enable shuffling for the training split after it is defined. Fit any normalization using training data only. Constructing a dataset from a split frame restarts indicator history at that split boundary, which may reduce sample counts for long indicators or GARCH.

Dataset construction holds the selected feature matrix and eligible window positions in memory. Each item slices its lookback on demand. Check per-stock counts before batching, especially when selecting SMA100 or GARCH.

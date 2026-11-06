# Stock forecasting pipeline

This folder turns minute OHLCV data into validated Parquet features and PyTorch training windows.

| Location | Contents |
| --- | --- |
| [datasets/NIFTY50](datasets/NIFTY50/) | Raw per-stock OHLCV CSVs |
| [eda](eda/) | Coverage notebook, market-minute calendar, and gap reports |
| [helpers](helpers/) | Feature engineering, dataset classes, and training utilities |
| [processed_datasets](processed_datasets/) | Per-stock Parquet files and validation reports |

| Guide | Start here for |
| --- | --- |
| [Examples](EXAMPLES.md) | Run the HYUNDAI training, evaluation, and profit simulation |
| [Transformation flow](docs/DATA_TRANSFORMATION_FLOW.md) | Inputs, rebuild command, and output checks |
| [Feature reference](docs/FEATURES.md) | Indicator definitions, imputation, and warm-up |
| [Data loaders](docs/DATA_LOADERS.md) | Feature selection, forecasting windows, and PyTorch batching |
| [Utilities](docs/UTILS.md) | Training, evaluation, and checkpoints |

## Train HYUNDAI

From inside this folder, run with NumPy, pandas, PyTorch, pyarrow, arch, scikit-learn, and matplotlib installed.

```powershell
python main.py --epochs 20
```

The entry point trains an LSTM on observed-bar windows, validates on later sessions, and evaluates the final held-out sessions. It saves the checkpoint, test predictions, trade log, and metrics in `runs/HYUNDAI/`. The profit simulation applies the calculator to eligible one-minute close-to-close returns with a fixed trading threshold and fee.

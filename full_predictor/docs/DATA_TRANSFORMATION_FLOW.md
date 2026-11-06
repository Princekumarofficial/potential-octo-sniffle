# Data transformation flow

1. Read stock CSVs from `datasets/NIFTY50/` and the minute calendar from `eda/eda_outputs/expected_market_minutes.csv`. Exclude IRFC, HDFCLIFE, and SBILIFE.
2. Sort timestamps, reject duplicates, and align each stock to the grid within its observed date range. Flag absent and invalid bars.
3. Fill eligible gaps of 1–3 minutes with the per-column OHLCV mean of equally many observed, valid bars on each side. The entire neighborhood must be consecutive and within one session. Other gaps stay NaN.
4. Calculate indicators and the daily GARCH forecast. Preserve indicator warm-up NaNs.
5. Validate and save one Parquet file per stock in `processed_datasets/NIFTY50/`. Reload each file and verify its values and null mask.

Two-sided fills can influence indicators after a gap. The forecasting datasets reconstruct features from observed bars. See [Data loaders](DATA_LOADERS.md) and the [feature reference](FEATURES.md).

## Rebuild and verify

From inside `full_predictor`, activate a Python environment with NumPy, pandas, arch, and pyarrow.

```powershell
python -m helpers.feature_engineering --workers 3
```

Use one worker to reduce memory use. The command replaces generated stock files and reports. The transformation cell in the [EDA notebook](../eda/eda.ipynb) also supports rebuilding.

In `processed_datasets/feature_validation.csv`, check the validation status and confirm that `missing_bars` equals `imputed_bars + remaining_missing_bars` for each stock. Review the NaN report for feature coverage and the manifest for the calendar fingerprint and imputation policy.

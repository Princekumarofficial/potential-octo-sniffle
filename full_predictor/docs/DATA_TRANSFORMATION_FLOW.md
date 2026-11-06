# Data transformation flow

1. **Read the inputs.** Load one OHLCV CSV per stock from `datasets/NIFTY50/`. Use the expected minute timestamps in `eda_outputs/expected_market_minutes.csv` as the reference calendar. Exclude IRFC, HDFCLIFE, and SBILIFE.
2. **Align each stock to the calendar.** Parse and sort timestamps, check for duplicates, and reindex from the stock’s first through last observed date. Keep only timestamps on the reference grid. Mark missing and invalid bars separately. Missing OHLCV values stay NaN.
3. **Build continuous segments.** Start a new segment at each session boundary, timestamp gap, or missing or invalid bar. This prevents minute returns and rolling indicators from spanning gaps.
4. **Calculate 35 features.** Within each segment, compute simple and log returns, SMA and EMA, RSI, Bollinger Bands, trend and slope, MACD, rolling minute volatility, and volume features. Fit a daily GARCH(1,1) model using prior daily returns and attach each session’s forecast to its valid minute bars. Keep warm-up values and unavailable forecasts as NaN.
5. **Validate and save.** Check timestamp order and uniqueness, NaN preservation, feature ranges, and return alignment. Write one Parquet file per stock to `processed_datasets/NIFTY50/`, then reload each file and verify that its values and missing entries match the transformed data.

The completed build contains **97 Parquet files, 87,398,865 rows, and 102,724 missing bars preserved as NaN**. The parent `processed_datasets/` folder contains the validation summary, per-feature NaN counts, GARCH fit audit, and run manifest. See [FEATURES.md](FEATURES.md) for formulas and rebuild instructions.

## Reproduce the build

1. From the `BTP` repository root, make sure the raw CSVs and saved minute calendar from step 1 are in place under `full_predictor/`. Use the same inputs and the dependency versions recorded in `full_predictor/processed_datasets/feature_manifest.json` to reproduce the existing build.
2. Run the transformation directly from PowerShell using the project environment.

   ```powershell
   .\.venv\Scripts\python.exe -m full_predictor.feature_engineering --workers 3
   ```

   The environment needs NumPy, pandas, `arch`, and `pyarrow`. Use `--workers 1` to reduce memory use. The command uses the saved calendar and replaces the generated Parquet files and reports.
3. Check `full_predictor/processed_datasets/feature_validation.csv`. With the same inputs, expect 97 rows marked `passed` and a total of 102,724 missing bars across the 97 stock files.

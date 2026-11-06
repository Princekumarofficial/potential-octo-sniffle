# BTP stock forecasting project

This repository contains stock forecasting experiments and a pipeline that turns minute-level OHLCV data into validated feature datasets. The current feature build covers 97 stocks.

## Documentation

| Guide                                                                 | What it covers                                              |
| --------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Data transformation flow](full_predictor/DATA_TRANSFORMATION_FLOW.md) | Raw CSVs to Parquet, including rebuild and validation steps |
| [Feature reference](full_predictor/FEATURES.md)                        | Indicator formulas, NaN handling, and daily GARCH forecasts |
| [Utility reference](full_predictor/UTILS.md)                           | Windowing, training, evaluation, and checkpoint helpers     |
|                                                                       |                                                             |
|                                                                       |                                                             |

## Folder structure

| Location                                                                    | Contents                                            |
| --------------------------------------------------------------------------- | --------------------------------------------------- |
| [`full_predictor/`](full_predictor/)                                       | Feature pipeline, utilities, and validation tests   |
| [`full_predictor/datasets/`](full_predictor/datasets/)                     | Raw stock CSVs                                      |
| [`full_predictor/eda/`](full_predictor/eda/)                               | EDA notebook, minute calendar, and coverage reports |
| [`full_predictor/processed_datasets/`](full_predictor/processed_datasets/) | Per-stock Parquet files and validation reports      |
|                                                                             |                                                     |

The root notebooks cover model training and evaluation. For the feature build, start with the transformation guide. The calendar currently lives in `full_predictor/eda/eda_outputs/`, while the rebuild command expects it in `full_predictor/eda_outputs/`. Copy `expected_market_minutes.csv` to the expected location before rebuilding.

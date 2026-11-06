# full_predictor/utils.py

Utilities for preparing time series, training PyTorch models, measuring predictions,
simulating trading strategies, and saving checkpoints. Import from the project root:

```python
from full_predictor import utils
```

Dependencies: NumPy, pandas, PyTorch, scikit-learn, and matplotlib.
This module does not define a forecasting model architecture or read CSV files.

## Cleanup and migration

There is one training entry point: `train_torch_model_dataloader`.

| Removed helper | Replacement |
| --- | --- |
| `train_torch_model` | Wrap tensors in `TensorDataset` and `DataLoader`, then use the retained trainer. |
| `train_on_val_data` | Pass the desired loader as `train_loader`, with `val_loader=None`. |
| `train_on_split` | Pass the desired loader directly and provide an optimizer. |
| `train_torch_model_with_val_data` | Call the retained trainer explicitly for each training stage. |

The replacement returns `train_losses`, `val_losses`, and `best_val_loss`;
old return keys such as `val_train_losses`, `losses`, and `final_loss` are removed.
The trainer uses the supplied loader's ordering; configure shuffling when building it.
Gradient clipping is available through `clip_grad_norm`.
Once a split is used for fitting, it is no longer an independent evaluation split.

Unused `product`, `TensorDataset`, and `SARIMAX` imports were removed.
The root-level `utils.py` is separate and was not changed.

## Data preparation and tensors

| Function | Purpose and return value |
| --- | --- |
| `prepare_data(data)` | Return a copy with rows containing missing values removed. |
| `split_data(data, train_fraction=0.8, validation_fraction=0.1)` | Return three DataFrames in existing row order; does not sort or scale. |
| `fit_standard_scaler(data, columns=None)` | Fit and return a `StandardScaler`. Supply training rows only. |
| `data_to_tensor(data, columns=None, dtype=..., device=None)` | Convert tabular data to a two-dimensional tensor. |
| `normalize_to_tensor(data, scaler, ...)` | Apply an already fitted scaler and return a tensor. |
| `denormalize_tensor(data, scaler, ...)` | Inverse-transform values and return a tensor. |
| `calculate_technical_indicators(data, target_column="INR")` | Return a sorted copy with moving averages, exponential averages, RSI, Bollinger bands, volatility, ATR proxies, and log returns. |

Indicator warm-up rows contain missing values; clean them before training.
Log returns require positive prices. ATR columns are close-price proxies, not OHLC ATR.
Tensor conversion helpers use NumPy internally: detach tensors and move them to CPU
before passing CUDA tensors or tensors requiring gradients.

## Windows and DataLoaders

`TimeSeriesWindowDataset(features, targets, window_size=30, offset=0,
target_size=1, stride=1, valid_indices=None, dates=None, device=None,
dtype=torch.float32)` slices windows on demand.

- Features and targets must have the same row count.
- For sample start `s`, inputs are `[s:s + window_size]` and targets start at
  `s + window_size + offset`.
- `offset=0` predicts the next observation; `offset=-1` aligns the target with
  the final input observation. Avoid shifting targets twice.
- Input batches have shape `(batch, window_size, feature_count)`.
- One target and one future step produce `(batch,)`; one target and multiple
  steps produce `(batch, target_size)`; multiple targets produce
  `(batch, target_size, target_count)`.
- `get_target_indices()`, `get_target_dates()`, and
  `get_input_window_dates(idx)` expose alignment metadata.

`create_time_series_dataloaders(data, target_column="INR", feature_columns=None,
...)` sorts a DataFrame by index, makes chronological 80%/10%/10% splits by
default, and fits optional feature and target scalers on training rows only.
Specify feature columns explicitly to include historical target values as inputs;
the default excludes target columns.

The result contains `train_loader`, `val_loader`, `test_loader`, the three
corresponding datasets, scalers, column names, dates, raw/scaled arrays, and window
settings. A split with no valid windows has a `None` loader. Validation and test
windows may use preceding history, but their target windows stay within the split.
`stride` subsamples training windows only; validation and test use stride 1.
Prefer CPU datasets (`device=None`) and `num_workers=0`; the trainer moves batches
to the model's device.

## Training example

This example uses an existing numeric DataFrame named `df` with a chronological
index and a `Close` column. CSV loading belongs in `data_loaders.py`.

```python
import torch
from torch import nn
from full_predictor import utils

loaders = utils.create_time_series_dataloaders(
    utils.prepare_data(df),
    target_column="Close",
    feature_columns=["Close"],
    window_size=30,
    batch_size=32,
)
model = nn.Sequential(nn.Flatten(), nn.Linear(30, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = utils.train_torch_model_dataloader(
    model, optimizer, nn.MSELoss(),
    train_loader=loaders["train_loader"],
    val_loader=loaders["val_loader"],
    epochs=50,
    early_stopping_patience=5,
    clip_grad_norm=1.0,
)
```

The trainer modifies the model in place and returns loss history. With early
stopping enabled, it restores the best validation weights even if all epochs run.
`best_val_loss` is the minimum observed validation loss, or `None` without validation.
Epoch losses are averages of batch losses, not weighted by batch size.
Pass a scheduler through `scheduler`; plateau schedulers use validation loss when
available and training loss otherwise. Set `log_every=0` for silent operation.

`BigMoveAwareLoss(dir_penalty=0.5, mag_penalty=2.0, sigma_threshold=1.5)` weights
squared errors by target magnitude and adds a sign-mismatch term. Its thresholds
are in target units, including scaled units if targets were normalized. The sign
term uses `torch.sign` and does not provide a useful gradient through the sign.

## Evaluation and baselines

| Function | Behavior |
| --- | --- |
| `calculate_directional_metrics(predictions, targets, previous_values=None)` | Regression errors and directional counts, accuracy, precision, recall, and F1. Percentage metrics use 0–100 units. |
| `evaluate_predictions(...)` | Compact subset: MAE, MSE, RMSE, directional accuracy, correct days, and total days. |
| `evaluate_torch_model(model, X_test, y_test, ..., plot=True)` | Evaluate full tensors, print errors, optionally plot, and return `(predictions, metrics)`. Use CPU model/tensors for its NumPy conversions. |
| `plot_predictions_vs_targets(predictions, targets, x_values=None, ...)` | Display a matplotlib comparison plot. |
| `predict_naive(data, target_column="INR")` | Previous-row baseline; fills the first missing value with the full-series mean. |
| `predict_moving_average(data, window_size=3, target_column="INR")` | Rolling mean including the current row; fills warm-up values with the full-series mean. |

For price directions, pass the previous observed prices as `previous_values`.
For return-sign accuracy, pass an equally sized array of zeros. Without this
argument, direction is measured relative to the preceding target and the first
observation is excluded. Flat movements count as up. Supply aligned arrays of
equal length; the current metric implementation truncates unequal inputs.

The baseline mean fills use future observations, and the moving average includes
the current observation. These existing helpers should not be treated as
leakage-free forecasting benchmarks without correcting their alignment/fill policy.

## Trading simulation and optimization

- `AdvancedProfitCalculator(predictions, actual_returns, ...)` simulates a strategy
  with position sizing, fees, spread, thresholds, and stop/take-profit settings.
  Use `summary()` for printed statistics and `plot()` for charts.
- `GeneticStrategyOptimizer(predictions, actual_returns, ...)` searches strategy
  parameters. `optimize()` returns the best parameters, fitness, calculator,
  history, and optimizer; `plot_convergence()` plots progress.
- `optimize_strategy_ga(...)` is a convenience constructor plus `optimize()` call.
  It is retained because it provides the public one-call optimization interface.

Use aligned, original-scale returns for these APIs. Optimize strategy settings on
development data and assess the chosen settings on a separate held-out period.
See class docstrings for the individual trading parameters.

## Results and checkpoints

- `save_model_results_to_json(model_results, output_path="model_results.json")`
  serializes common NumPy/pandas/PyTorch values and returns the output path.
  Create the parent directory first.
- `store_model_prediction(model_results, model_name, predictions, results, split,
  ...)` updates the supplied dictionary, writes it to JSON, and returns the model entry.
- `export_model(model, filepath, ..., loaders_dict=None)` writes weights, inferred
  architecture settings, scalers, feature order, window size, and metadata.
  Pass `trained_till` explicitly; its inherited default is `"test"`.
- `load_model(filepath, model_class=YourModel, ...)` requires the model class and
  returns the model, scalers, metadata, and a `predict` callable. The architecture
  must accept the saved constructor settings; automatic configuration is tailored
  to models exposing `lstm` and optionally `fc`, not arbitrary PyTorch modules.

```python
# YourForecastModel is the same compatible class used to build trained_model.
utils.export_model(
    trained_model, "full_predictor/checkpoints/model.pt",
    trained_till="train", loaders_dict=loaders,
)
loaded = utils.load_model(
    "full_predictor/checkpoints/model.pt", model_class=YourForecastModel,
)
prediction = loaded["predict"](df.tail(loaders["window_size"]))
```

The prediction helper scales 2D raw input and selects its final window. Its 3D
batch input is assumed already scaled. It flattens predictions and is intended
for single-target inference. Checkpoints contain pickled scaler objects and must
come from a trusted source.

Private helpers `_to_numpy` and `_to_jsonable` support the public functions above.

# NIFTY 50 forecasting notebook

`nifty_50.ipynb` prepares five-minute market data for a model that predicts the next observed NIFTY 50 bar's return. It combines NIFTY 50, INDIA VIX, and NIFTY BANK features and produces PyTorch training, validation, and test dataloaders.

The notebook implements data exploration, feature engineering, target construction, sequence preparation, and a trainable bidirectional LSTM regressor. Its final cell runs training and plots training/validation loss. The test section compares predicted and actual log returns and reports directional correctness.

## 1. Load and inspect the data

Run the notebook from the project root with pandas, NumPy, Matplotlib, and PyTorch available in its Python environment. It reads:

| File under `FRED/` | Purpose |
| --- | --- |
| `NIFTY 50_5minute (1).csv` | Main timeline, input features, and forecast target |
| `INDIA VIX_5minute.csv` | Additional volatility-index features |
| `NIFTY BANK_5minute.csv` | Additional banking-index features |

Each source needs `date`, `open`, `high`, `low`, and `close`. The notebook converts dates, displays dataframe information, and plots closing prices. Indicator construction sorts each source chronologically and rejects missing or duplicate timestamps.

The recent-data chart uses a fixed cutoff of `2026-04-01` and filters weekdays only for that plot. This does not restrict the training data. The VIX plot currently has a NIFTY 50 title, although its plotted values come from VIX.

## 2. Encode time

`get_sine_transformed_time()` adds sine/cosine pairs for time within the trading session and day of the week:

- Session phase uses 09:15 to 15:30.
- Weekday phase uses the weekday number divided by seven.
- Both sine and cosine are retained to represent position around the cycle.

Timestamps are assumed to already represent Indian market local time; the function does not convert timezones or filter out-of-session bars. Only one set of calendar encodings is included in the final inputs, from NIFTY 50.

## 3. Compute technical features

`add_technical_indicators()` can be applied separately to each source. In the current notebook, it is called for NIFTY 50 and NIFTY BANK; INDIA VIX currently contributes OHLC only. Each technical feature uses the current completed bar and/or earlier bars.

| Group | Features and lookbacks |
| --- | --- |
| Returns | One-bar log return; simple returns over 1, 3, 6, 12, 24, and 75 bars |
| Lagged returns | One-bar return lagged by 1, 2, 3, 6, and 12 bars |
| Trend | SMA and EMA over 5, 12, 20, 26, 50, and 75 bars; close relative to each average |
| MACD | EMA-12 minus EMA-26, nine-span signal, histogram, and price-relative MACD |
| Momentum | Wilder-smoothed RSI-14; stochastic K-14 and D-3 |
| Volatility | Wilder-smoothed ATR-14 and price-relative ATR; log-return standard deviation over 12, 20, and 75 bars |
| Bands | Bollinger upper/lower bands, width, position, and price z-score over 20 bars |
| Candle structure | Body, range, upper/lower wicks, and close position within the range |
| Gaps and breakouts | Open relative to previous close; close relative to the previous 20-bar high and low |

OHLC prices are also retained. Volume is excluded because index volume may be unavailable or zero. Warm-up values and undefined ratios remain missing; they are never backfilled. Rolling windows count observed bars and continue across trading sessions rather than resetting each day. Return volatility is per bar, not annualized.

## 4. Align VIX and NIFTY BANK

The sorted NIFTY 50 dataframe is the master timeline. Exact timestamp left joins attach auxiliary features with prefixes such as `vix_close`, `vix_rsi_14`, `bank_close`, and `bank_return_1`.

The joins preserve every NIFTY 50 row. A missing auxiliary timestamp leaves missing features; a sequence containing those features is excluded later. There is no forward fill, backward fill, nearest-time matching, or removal of rows that would join separated observations together.

Numeric features are selected automatically, excluding volume, raw time/day fields, and names starting with `target`. Auxiliary calendar encodings are omitted to avoid duplication. With the current indicator calls and source columns, there are **138 features: 69 from NIFTY 50, 4 from VIX, and 65 from NIFTY BANK**. Applying `add_technical_indicators(india_vix_df)` before the loader cells would also add VIX indicators, giving 199 features. Use `len(feature_columns)` as the model input size rather than hardcoding this count.

All three sources must use compatible timestamps and bar definitions. Forecasting assumes the current bars from all three sources have finished and are available.

## 5. Define the prediction target

Let `C[j]` be NIFTY 50's close at row `j`. The inputs for target row `j` end at row `j - 1`.

| `TARGET_KIND` | Target at row `j` |
| --- | --- |
| `"log_return"` (default) | `log(C[j] / C[j - 1])` |
| `"return"` | `C[j] / C[j - 1] - 1` |

Simple returns are fractions: `0.01` means 1%. Targets are not standardized. NIFTY 50 closes must be finite and positive.

“Next” means the next observed NIFTY 50 row, not necessarily exactly five minutes later. A target can span an overnight break, weekend, or missing source bar. Targets are computed before invalid input windows are excluded, preserving this definition.

## 6. Split and scale

Chronological boundaries are calculated from the full NIFTY 50 timeline:

- First 70% of rows: training target period.
- Next 15%: validation target period.
- Remaining 15%: test target period.

Samples are assigned by their **target row**, not randomly. Excluding invalid windows means final sample counts need not be exactly 70/15/15.

For each feature, the notebook calculates the mean and population standard deviation using fully finite rows in the training period only. Every split then uses those same statistics:

```text
scaled_feature = (feature - training_mean) / training_std
```

Standard deviations below `1e-12` are replaced with 1. Inputs are converted to float32. A prefix count of invalid rows efficiently rejects any window containing a nonfinite feature, without compressing the timeline.

Validation/test windows can include earlier historical context from preceding splits. This is intentional: those observations are available before the target occurs. Target periods remain separate, but adjacent windows can share input bars. The notebook does not implement purged splits or an embargo.

## 7. Construct shifted sequences and batches

`NextBarWindowDataset` stores shared feature/target tensors and a list of eligible target indices. It slices a sequence when requested instead of materializing all overlapping windows in memory:

```python
X = features[j - SEQ_LEN:j]
y = targets[j:j + 1]
```

For a small illustrative sequence length of 3:

```text
Input rows 0, 1, 2  -> return from close 2 to close 3
Input rows 1, 2, 3  -> return from close 3 to close 4
Input rows 2, 3, 4  -> return from close 4 to close 5
```

These examples assume all input rows are valid. With the default stride of 1, every eligible target is used. `WINDOW_STRIDE` selects every nth **eligible training target**; after invalid windows are excluded, this need not mean exactly n original bars between selected targets. Validation and test always use all eligible targets.

Training uses `shuffle=True`, so each epoch visits the same eligible training windows in a newly shuffled order. It does not generate new random window lengths or randomly jitter a fixed sample. Time order inside every window stays intact. Reuse the same loader across epochs without reseeding its generator. Shifted windows expose the model to different contexts, but do not by themselves guarantee improved generalization.

| Setting | Default | Meaning |
| --- | --- | --- |
| `SEQ_LEN` | `75` | Completed input bars per sample, roughly one trading session |
| `BATCH_SIZE` | `64` | Samples per batch |
| `WINDOW_STRIDE` | `1` | Subsampling step over eligible training targets |
| `TARGET_KIND` | `"log_return"` | Target formula |
| `TRAIN_FRACTION` | `0.70` | Training timeline fraction |
| `VAL_FRACTION` | `0.15` | Validation timeline fraction |
| `SEED` | `42` | Training sampler seed; does not seed model initialization |

All loaders use `num_workers=0` and `drop_last=False`, so the final batch can be smaller. Validation and test use `shuffle=False`.

Default full-batch shapes are:

```text
X: [64, 75, 138] = [batch, sequence, features]
y: [64, 1]      = [batch, next-bar return]
```

## 8. Checks and model integration

Run the notebook from top to bottom. After changing loader configuration, rerun all cells under `DATA LOADERS`. After changing data or indicators, rerun the upstream preparation cells too.

The final cells print split sizes, target date ranges, feature count, missing auxiliary-close counts, and batch shapes. They assert that an inspected training batch is finite and that its shapes are correct, and verify the first training sample's target against the corresponding price ratio. Inspecting a batch advances the training sampler's random generator; subsequent epochs still shuffle normally.

The model consumes `train_loader` and `val_loader`, accepting `[batch, sequence, len(feature_columns)]` and producing `[batch, 1]`. `test_loader` is reserved for final evaluation. The earlier commented loop illustrates the interface; the model section now provides an executable training function.

Because training batches contain shuffled windows, a recurrent model should normally reset its hidden state between independent windows rather than carry state across batches. Use validation for model selection and reserve test targets for final evaluation.

## 9. Convert returns back to prices

`return_to_next_close()` converts predicted returns into a next-close forecast:

```text
Simple return: predicted_close = last_input_close * (1 + predicted_return)
Log return:    predicted_close = last_input_close * exp(predicted_log_return)
```

For predictions collected in order from `test_loader`:

```python
predicted_closes = return_to_next_close(
    closes[test_indices - 1],
    test_predictions.reshape(-1),
    target_kind=TARGET_KIND,
)
actual_closes = closes[test_indices]
target_dates = forecast_df.loc[test_indices, "date"]
```

Here `test_predictions` should be a NumPy array; convert a PyTorch tensor with `.detach().cpu().numpy()` first. Flatten predictions to avoid accidental broadcasting against the one-dimensional close array.

For future inference, preserve `feature_columns` in the same order, `feature_mean`, `feature_std`, `SEQ_LEN`, and `TARGET_KIND` alongside the trained model. The current loaders contain labeled historical samples; constructing the latest unlabeled inference window and saving model artifacts are not yet implemented.


## 10. BiLSTM, loss, optimizer, and training

`BiLSTMRegressor` uses one bidirectional LSTM layer with 64 hidden units per direction by default. It concatenates the final forward and backward hidden states of the last layer, producing a 128-value representation of the historical window. The backward direction only reads the supplied historical window; it never reads the future target bar.

The dense regression head is `Linear(128, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)`. Its output is multiplied by the standard deviation of eligible training targets, returning unrestricted raw returns. The same scale normalizes the regression loss; no mean is subtracted, so zero remains the direction boundary. LSTM inter-layer dropout is enabled only when `NUM_LAYERS > 1`; head dropout applies with either setting.

The initialization cell selects CUDA when available, otherwise CPU, seeds model initialization, and builds:

- `loss_fn`: `ReturnDirectionLoss`, combining scale-normalized MSE and a differentiable direction penalty (weight 5.0).
- `optimizer`: Adam with learning rate `1e-3`.
- `model`: input size derived from `len(feature_columns)`.

`train_model()` defaults to 20 epochs, patience of 5 epochs, and a gradient-norm limit of 1.0. Each epoch trains on shuffled batches, then evaluates validation data without gradients or dropout. Epoch loss is weighted by sample count, so a smaller final batch is handled correctly. The helper rejects mismatched output/target shapes, nonfinite losses, and nonfinite gradient norms.

Whenever validation loss improves, both model and optimizer states are copied in memory. After early stopping or the epoch limit, the best states are restored and the model is placed in evaluation mode. The return dictionary contains `history`, `best_epoch`, `best_val_loss`, and `stopped_early`. No checkpoint file is saved automatically.

Run the final training cell to fit the model and plot the loss curves. Rerun initialization for a fresh model/optimizer; rerunning only training continues from the restored state. Training loss includes dropout and changing parameters, so it is not directly equivalent to validation loss measured with fixed parameters and dropout disabled. This is a simple baseline; low validation loss alone does not establish useful predictive performance. The test section now reports regression errors, a zero-return RMSE baseline, and directional correctness.


## 11. Test log returns and direction

Run `TEST THE MODEL` after training. `evaluate_test_model()` evaluates the current model with gradients and dropout disabled, then restores its prior training/evaluation mode. It requires log-return targets and an ordered test loader with `drop_last=False`. Test metrics are calculated independently of the configured training loss.

`test_results` contains each target timestamp, previous input close, actual close, actual and predicted log returns, signed/absolute errors, actual/predicted direction, and a `direction_correct` flag. Target values are checked against the corresponding close-price ratios. Direction has three classes: negative = down, exact zero = flat, positive = up. Correctness requires an exact class match; flat observations are included in the overall accuracy denominator.

`test_metrics` reports MSE, RMSE, MAE, zero-return baseline RMSE, directional accuracy, and correct/incorrect counts. `direction_summary` breaks performance down by actual direction, and `confusion_matrix` has actual directions as rows and predicted directions as columns. Full timestamped subsets are available in `correct_predictions` and `incorrect_predictions`; the notebook previews the first ten of each.

The plots compare actual/predicted returns over the latest 300 eligible test targets, mark actual returns green/red according to directional correctness, and show an actual-versus-predicted scatter over the entire test set. Change `PLOT_LAST_N` to inspect a different recent interval. All metrics use the complete eligible test set, regardless of the plotting interval. These cells evaluate the current model; they do not load a checkpoint or train a new model.


## 12. Avoiding an always-up solution

The old `BigMoveAwareLoss` used `sign(prediction)`, whose gradient is zero. Its direction penalty could influence checkpoint selection but supplied no corrective gradient. `ReturnDirectionLoss` instead uses `softplus(-sign(target) * prediction / RETURN_SCALE)`, combined with normalized MSE. Zero targets are excluded from the directional term and retained in regression. `RETURN_SCALE` is fitted only to eligible training targets and stored in the model as a buffer; test outputs remain raw log returns.

Validation history now includes directional accuracy, predicted up/down percentages, and the most frequent actual direction's share as a majority-class reference. This reference is descriptive, not a deployable rule selected from validation labels. Check these alongside loss to detect one-sided predictions. The loss plot contains only training and validation combined losses, not accuracy percentages.

Rerun the loss definition, model/optimizer initialization, training function, training cell, and test cells. Previous model-section outputs were cleared because they describe the old objective. The new objective was checked for corrective gradients and for learning both signs on a synthetic learnable sequence task; this does not establish better NIFTY performance. Do not force a 50/50 output split: real improvement requires validation performance above simple baselines, followed by held-out evaluation. Tune direction weight on validation only.


The directional weight is now `5.0` (previously `0.5`): loss = normalized MSE + 5 × directional softplus. This increases the directional contribution and its gradient tenfold at fixed predictions, before gradient clipping. Retrain from initialization; combined loss values are not comparable across different weights. A stronger penalty can sacrifice return-magnitude accuracy and does not guarantee predictions of both signs or improved validation accuracy.

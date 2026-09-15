import json
from itertools import product
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler


FRED_FILES = {
    "INR": "DEXINUS.csv",
    "JPY": "DEXJPUS.csv",
    "EUR": "DEXUSEU.csv",
    "FFER": "DFF.csv",
    "US10YY": "DGS10.csv",
}


def load_fred_data(data_dir="FRED", file_map=None):
    """Load the configured FRED series and combine them by date."""
    files = file_map or FRED_FILES
    frames = {
        name: pd.read_csv(
            f"{data_dir}/{filename}",
            index_col=0,
            header=0,
            parse_dates=True,
        )
        for name, filename in files.items()
    }
    data = pd.concat(frames.values(), axis=1)
    data.columns = list(frames)
    return data.sort_index()


def prepare_data(data):
    """Remove rows with missing values from a combined dataset."""
    return data.dropna().copy()


def _to_numpy(data, columns=None):
    """Convert a DataFrame, mapping, Series, or array to a 2D NumPy array."""
    if isinstance(data, pd.DataFrame):
        values = data.loc[:, columns] if columns is not None else data
        return values.to_numpy(dtype=np.float32)
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float32).reshape(-1, 1)
    if isinstance(data, Mapping):
        keys = list(columns) if columns is not None else list(data)
        return np.column_stack([np.asarray(data[key]) for key in keys]).astype(np.float32)

    values = np.asarray(data, dtype=np.float32)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def fit_standard_scaler(data, columns=None):
    """Fit a feature-wise StandardScaler and return it."""
    scaler = StandardScaler()
    scaler.fit(_to_numpy(data, columns))
    return scaler


def data_to_tensor(data, columns=None, dtype=torch.float32, device=None):
    """Convert tabular data or a NumPy mapping to a 2D PyTorch tensor."""
    return torch.as_tensor(_to_numpy(data, columns), dtype=dtype, device=device)


def normalize_to_tensor(data, scaler, columns=None, dtype=torch.float32, device=None):
    """Scale data with a fitted scaler and convert the result to a tensor."""
    values = scaler.transform(_to_numpy(data, columns))
    return torch.as_tensor(values, dtype=dtype, device=device)


def denormalize_tensor(data, scaler, dtype=torch.float32, device=None):
    """Convert normalized tensor-like data back to its original scale."""
    values = scaler.inverse_transform(_to_numpy(data))
    return torch.as_tensor(values, dtype=dtype, device=device)


def split_data(data, train_fraction=0.8, validation_fraction=0.1):
    """Split a time series into ordered train, validation, and test frames."""
    train_size = int(len(data) * train_fraction)
    validation_size = int(len(data) * validation_fraction)
    return (
        data.iloc[:train_size].copy(),
        data.iloc[train_size:train_size + validation_size].copy(),
        data.iloc[train_size + validation_size:].copy(),
    )


def plot_dataset(data, title="Full Dataset", figsize=(14, 12)):
    axes = data.plot(
        subplots=True,
        figsize=figsize,
        sharex=True,
        layout=(data.shape[1], 1),
        title=title,
    )
    for axis in axes.flatten():
        axis.set_xlabel("Date")
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_targets(
    predictions,
    targets,
    x_values=None,
    title="Predictions vs Targets",
):
    """Plot predictions and targets against supplied timestamps or positions."""
    target_values = np.asarray(targets).reshape(-1)
    prediction_values = np.asarray(predictions).reshape(-1)
    if len(prediction_values) != len(target_values):
        raise ValueError("Predictions and targets must have the same length.")
    if x_values is None:
        x_values = np.arange(len(target_values))
    else:
        x_values = np.asarray(x_values)
        if len(x_values) != len(target_values):
            raise ValueError("x_values must have the same length as predictions and targets.")

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, target_values, label="Targets", color="blue")
    plt.plot(x_values, prediction_values, label="Predictions", color="orange")
    plt.title(title)
    plt.xlabel("Date" if x_values.dtype.kind == "M" else "Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_directional_metrics(predictions, targets, previous_values=None):
    """Compute directional metrics (UP/DOWN accuracy, precision, recall, F1) and regression metrics.

    Direction is defined as UP (+1) if price moves up, DOWN (-1) if it moves down.
    A prediction is correct if its predicted movement direction matches the actual movement direction.
    """
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targs = np.asarray(targets, dtype=np.float64).reshape(-1)

    min_len = min(len(preds), len(targs))
    preds = preds[-min_len:]
    targs = targs[-min_len:]

    if previous_values is not None:
        prev = np.asarray(previous_values, dtype=np.float64).reshape(-1)[-min_len:]
        actual_diff = targs - prev
        pred_diff = preds - prev
    else:
        # Base price is the previous day's actual target
        actual_diff = np.diff(targs)
        pred_diff = preds[1:] - targs[:-1]

    # Directions: +1 for UP, -1 for DOWN, 0 for FLAT
    actual_direction = np.sign(actual_diff)
    # Convert any 0 changes to +1 (or default to positive movement if zero diff)
    actual_direction[actual_direction == 0] = 1

    pred_direction = np.sign(pred_diff)
    pred_direction[pred_direction == 0] = 1

    is_correct = (actual_direction == pred_direction)
    correct_days = int(np.sum(is_correct))
    total_days = int(len(is_correct))
    directional_accuracy = float(correct_days / total_days * 100.0) if total_days > 0 else 0.0

    # UP and DOWN counts
    actual_up = (actual_direction > 0)
    actual_down = (actual_direction < 0)
    pred_up = (pred_direction > 0)
    pred_down = (pred_direction < 0)

    true_up = int(np.sum(actual_up & pred_up))
    false_up = int(np.sum((~actual_up) & pred_up))
    true_down = int(np.sum(actual_down & pred_down))
    false_down = int(np.sum((~actual_down) & pred_down))

    up_precision = float(true_up / (true_up + false_up) * 100.0) if (true_up + false_up) > 0 else 0.0
    up_recall = float(true_up / int(np.sum(actual_up)) * 100.0) if np.sum(actual_up) > 0 else 0.0
    up_f1 = float(2 * up_precision * up_recall / (up_precision + up_recall)) if (up_precision + up_recall) > 0 else 0.0

    down_precision = float(true_down / (true_down + false_down) * 100.0) if (true_down + false_down) > 0 else 0.0
    down_recall = float(true_down / int(np.sum(actual_down)) * 100.0) if np.sum(actual_down) > 0 else 0.0
    down_f1 = float(2 * down_precision * down_recall / (down_precision + down_recall)) if (down_precision + down_recall) > 0 else 0.0

    errors = preds - targs
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(errors / (targs + 1e-8))) * 100.0)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "correct_days": correct_days,
        "incorrect_days": total_days - correct_days,
        "total_days": total_days,
        "actual_up_days": int(np.sum(actual_up)),
        "actual_down_days": int(np.sum(actual_down)),
        "predicted_up_days": int(np.sum(pred_up)),
        "predicted_down_days": int(np.sum(pred_down)),
        "up_precision": up_precision,
        "up_recall": up_recall,
        "up_f1": up_f1,
        "down_precision": down_precision,
        "down_recall": down_recall,
        "down_f1": down_f1,
        "is_correct_direction": is_correct,
        "actual_direction": actual_direction,
        "predicted_direction": pred_direction,
    }


def evaluate_predictions(predictions, targets, previous_values=None):
    """Evaluate predictions returning both standard regression and directional metrics."""
    metrics = calculate_directional_metrics(predictions, targets, previous_values)
    return {
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "directional_accuracy": metrics["directional_accuracy"],
        "correct_days": metrics["correct_days"],
        "total_days": metrics["total_days"],
    }


def get_target_data(data_dir="FRED", split=None):
    """Load target FRED series and return train/validation/test target splits."""
    data = prepare_data(load_fred_data(data_dir))
    data["TARGET_NEXT"] = data["INR"].shift(-1)
    clean_data = data.dropna(subset=["TARGET_NEXT"])

    train_df, val_df, test_df = split_data(clean_data, train_fraction=0.8, validation_fraction=0.1)

    splits = {
        "train": train_df["TARGET_NEXT"],
        "validation": val_df["TARGET_NEXT"],
        "test": test_df["TARGET_NEXT"],
        "train_df": train_df,
        "validation_df": val_df,
        "test_df": test_df,
        "full_df": clean_data,
    }
    if split is not None:
        return splits.get(split, clean_data["TARGET_NEXT"])
    return splits


def plot_directional_candles(
    predictions,
    targets,
    dates=None,
    model_name="Model",
    split="test",
    title=None,
    figsize=(16, 7),
    zoom_range=None,
    correct_color="#1E88E5",
    incorrect_color="#FF4081",
):
    """Plot target series and predictions with blue (correct) and pink (incorrect) candles or boxes."""
    from matplotlib.patches import Patch, Rectangle

    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targs = np.asarray(targets, dtype=np.float64).reshape(-1)
    min_len = min(len(preds), len(targs))
    preds = preds[-min_len:]
    targs = targs[-min_len:]

    metrics = calculate_directional_metrics(preds, targs)
    is_correct = metrics["is_correct_direction"]

    if dates is not None:
        x_dates = pd.to_datetime(np.asarray(dates)[-min_len:])
    else:
        x_dates = np.arange(min_len)

    # If zoom_range is requested (e.g. integer slice or date slice)
    if zoom_range is not None:
        if isinstance(zoom_range, int):
            start_idx = max(0, min_len - zoom_range)
            sl = slice(start_idx, min_len)
        elif isinstance(zoom_range, (tuple, list)) and len(zoom_range) == 2:
            sl = slice(zoom_range[0], zoom_range[1])
        else:
            sl = slice(None)
    else:
        sl = slice(None)

    sub_targs = targs[sl]
    sub_preds = preds[sl]
    sub_x = x_dates[sl]

    # Calculate candle base from previous targets
    # For day i in sub slice:
    indices = np.arange(min_len)[sl]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual and predicted lines
    ax.plot(sub_x, sub_targs, label="Actual Target (INR)", color="#0D47A1", linewidth=1.8, alpha=0.9, zorder=3)
    ax.plot(sub_x, sub_preds, label="Predicted (INR)", color="#FF8F00", linewidth=1.5, linestyle="--", alpha=0.9, zorder=4)

    # Draw boxes/candles for each day
    # Box spans between previous actual price and current actual price (or prediction spread)
    box_width = 0.6 if dates is None or not hasattr(sub_x, 'dtype') or sub_x.dtype.kind != 'M' else pd.Timedelta(days=0.8)

    for i in indices:
        if i == 0:
            continue
        curr_x = x_dates[i]
        prev_y = targs[i - 1]
        curr_y = targs[i]
        box_bottom = min(prev_y, curr_y)
        box_height = abs(curr_y - prev_y)
        if box_height == 0:
            box_height = 0.02  # minimal height for visibility

        correct = is_correct[i - 1]
        c = correct_color if correct else incorrect_color

        if hasattr(curr_x, 'to_pydatetime'):
            # Datetime box
            ax.bar(
                curr_x,
                box_height,
                bottom=box_bottom,
                width=box_width,
                color=c,
                alpha=0.45,
                edgecolor=c,
                linewidth=1.0,
                zorder=2,
            )
        else:
            # Numeric x box
            rect = Rectangle(
                (curr_x - 0.35, box_bottom),
                0.7,
                box_height,
                facecolor=c,
                alpha=0.45,
                edgecolor=c,
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(rect)

    # Subtitle with metrics
    plot_title = title or f"{model_name} Directional Evaluation ({split.title()} Split)"
    stats_text = (
        f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%  "
        f"({metrics['correct_days']}/{metrics['total_days']} days correct) | "
        f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}"
    )
    ax.set_title(f"{plot_title}\n{stats_text}", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Date" if hasattr(sub_x, "dtype") and sub_x.dtype.kind == "M" else "Time Steps", fontsize=11)
    ax.set_ylabel("INR Value", fontsize=11)

    legend_elements = [
        plt.Line2D([0], [0], color="#0D47A1", lw=2, label="Actual Target (INR)"),
        plt.Line2D([0], [0], color="#FF8F00", lw=2, linestyle="--", label="Predicted (INR)"),
        Patch(facecolor=correct_color, edgecolor=correct_color, alpha=0.5, label=f"Correct Direction (Blue: {metrics['correct_days']} days)"),
        Patch(facecolor=incorrect_color, edgecolor=incorrect_color, alpha=0.5, label=f"Incorrect Direction (Pink: {metrics['incorrect_days']} days)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    plt.show()
    return fig, ax



def _to_jsonable(value):
    """Convert common scientific Python values into JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_results_to_json(model_results, output_path="model_results.json"):
    """Save model metrics and predictions to a readable JSON file."""
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(_to_jsonable(model_results), indent=2),
        encoding="utf-8",
    )
    return output_path


def plot_model_metrics(model_results, split="test", log_scale=False, targets=None):
    """Plot MAE, MSE, and RMSE comparisons across models for a data split."""
    if isinstance(model_results, pd.DataFrame):
        df = model_results
        included_names = list(df["model_name"])
        mae_values = list(df["mae"]) if "mae" in df.columns else []
        mse_values = list(df["mse"]) if "mse" in df.columns else []
        rmse_values = list(df["rmse"]) if "rmse" in df.columns else [np.sqrt(v) for v in mse_values]
    else:
        model_names = list(model_results)
        included_names = []
        mae_values = []
        mse_values = []
        rmse_values = []

        target_series = targets[split] if isinstance(targets, dict) and split in targets else targets

        for model_name in model_names:
            result = model_results[model_name]
            predictions = result.get("predictions", {}).get(split)
            if target_series is not None and predictions is not None:
                m = calculate_directional_metrics(predictions, target_series)
                included_names.append(model_name)
                mae_values.append(m["mae"])
                mse_values.append(m["mse"])
                rmse_values.append(m["rmse"])
            else:
                metrics = result.get("metrics", result)
                split_metrics = metrics.get(split)
                if split_metrics is None:
                    continue
                included_names.append(model_name)
                mae = split_metrics.get("mae", 0.0)
                mse = split_metrics.get("mse", 0.0)
                rmse = split_metrics.get("rmse", np.sqrt(mse) if mse > 0 else 0.0)
                mae_values.append(mae)
                mse_values.append(mse)
                rmse_values.append(rmse)

    if not included_names:
        raise ValueError(f"No model metrics found for split: {split}")

    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics_data = [
        ("MAE", mae_values, "#1976D2", "Mean Absolute Error"),
        ("MSE", mse_values, "#F57C00", "Mean Squared Error"),
        ("RMSE", rmse_values, "#388E3C", "Root Mean Squared Error"),
    ]

    for axis, (metric_name, values, color, full_name) in zip(axes, metrics_data):
        bars = axis.bar(included_names, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
        axis.set_title(f"{full_name} ({metric_name})", fontsize=12, fontweight="bold")
        axis.set_ylabel(f"{metric_name} (lower is better)", fontsize=11)
        if log_scale:
            axis.set_yscale("log")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, linestyle=":", alpha=0.6, axis="y")

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                label_text = f"{height:.4f}" if height < 10 else f"{height:.2f}"
                axis.annotate(
                    label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    figure.suptitle(f"Regression Metrics Comparison Across Models ({split.title()} Split)", fontsize=14, fontweight="bold")
    figure.tight_layout()
    plt.show()
    return figure, axes



def predict_naive(data, target_column="INR"):
    return data[target_column].shift(1).fillna(data[target_column].mean())


def predict_moving_average(data, window_size=3, target_column="INR"):
    return data[target_column].rolling(window=window_size).mean().fillna(data[target_column].mean())


def grid_search_arima(train_data, validation_data, feature_columns, p_values, d_values, q_values, target_column="INR"):
    results = []
    best_order = None
    best_mae = float("inf")

    for order in product(p_values, d_values, q_values):
        try:
            candidate = SARIMAX(
                endog=train_data[target_column].to_numpy(),
                exog=train_data[feature_columns].to_numpy(),
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            forecast = candidate.forecast(
                steps=len(validation_data),
                exog=validation_data[feature_columns].to_numpy(),
            )
            mae = evaluate_predictions(forecast, validation_data[target_column])["mae"]
            results.append({"order": order, "mae": mae, "aic": candidate.aic})
            if mae < best_mae:
                best_order, best_mae = order, mae
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_order is None:
        raise RuntimeError("No ARIMA configuration fitted successfully.")
    return pd.DataFrame(results).sort_values("mae"), best_order


def fit_arima(data, feature_columns, order, target_column="INR"):
    return SARIMAX(
        endog=data[target_column].to_numpy(),
        exog=data[feature_columns].to_numpy(),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def forecast_arima(model, data, feature_columns, target_column="INR"):
    forecast = model.forecast(
        steps=len(data),
        exog=data[feature_columns].to_numpy(),
    )
    return forecast, evaluate_predictions(forecast, data[target_column])


def train_xgboost(model, train_data, validation_data, test_data, feature_columns, target_column="INR"):
    model.fit(
        train_data[feature_columns],
        train_data[target_column],
        eval_set=[
            (train_data[feature_columns], train_data[target_column]),
            (validation_data[feature_columns], validation_data[target_column]),
            (test_data[feature_columns], test_data[target_column]),
        ],
        verbose=100,
    )
    return model


def train_torch_model(model, optimizer, loss_function, X_train, y_train, X_validation, y_validation, epochs=100, log_every=10):
    train_losses = []
    validation_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(X_validation), y_validation)
        validation_losses.append(validation_loss.item())

        if log_every and (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {validation_loss.item():.4f}")
    return train_losses, validation_losses


def evaluate_torch_model(model, X_test, y_test, title="Model Predictions vs Actual Values", plot=True):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    metrics = evaluate_predictions(predictions, y_test)
    if plot:
        plot_predictions_vs_targets(predictions.numpy(), y_test.numpy(), title=title)
    print(f"Test MAE: {metrics['mae']:.4f}, Test MSE: {metrics['mse']:.4f}")
    return predictions, metrics


def compare_results(first, second, target, first_name="Model 1", second_name="Model 2"):
    print(f"{first_name} - MAE: {first['metrics']['mae']:.4f}, MSE: {first['metrics']['mse']:.4f}")
    print(f"{second_name} - MAE: {second['metrics']['mae']:.4f}, MSE: {second['metrics']['mse']:.4f}")
    plt.figure(figsize=(12, 6))
    plt.plot(np.asarray(first["predictions"]).reshape(-1), label=f"{first_name} Predictions")
    plt.plot(np.asarray(second["predictions"]).reshape(-1), label=f"{second_name} Predictions")
    plt.plot(np.asarray(target).reshape(-1), label="Actual Values")
    plt.title("Model Predictions Comparison")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_technical_indicators(data, target_column="INR"):
    result = data.sort_index().copy()
    series = result[target_column]
    result["MA_5"] = series.rolling(window=5).mean()
    result["MA_10"] = series.rolling(window=10).mean()
    result["MA_30"] = series.rolling(window=30).mean()
    result["MA_50"] = series.rolling(window=50).mean()
    result["MA_100"] = series.rolling(window=100).mean()
    result["EMA_5"] = series.ewm(span=5, adjust=False).mean()
    result["EMA_10"] = series.ewm(span=10, adjust=False).mean()
    result["EMA_30"] = series.ewm(span=30, adjust=False).mean()
    result["EMA_50"] = series.ewm(span=50, adjust=False).mean()
    result["EMA_100"] = series.ewm(span=100, adjust=False).mean()
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    relative_strength = gain.div(loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50)
    result["RSI"] = rsi
    rolling_mean = series.rolling(window=20).mean()
    rolling_std = series.rolling(window=20).std()
    result["BB_Middle"] = rolling_mean
    result["BB_Upper"] = rolling_mean + 2 * rolling_std
    result["BB_Lower"] = rolling_mean - 2 * rolling_std
    result["Volatility"] = rolling_std
    result["ATR_14_proxy"] = series.diff().abs().rolling(window=14).mean()
    result["ATR_%_proxy"] = result["ATR_14_proxy"] / series * 100
    result["Log_Returns"] = np.log(series / series.shift(1))
    return result

def store_model_prediction(
    model_results,
    model_name,
    predictions,
    results,
    split,
    output_path="model_results.json",
    metadata=None,
):
    """Store one model prediction and metrics split immediately in JSON."""
    model_entry = model_results.setdefault(
        model_name,
        {
            "model_name": model_name,
            "predictions": {},
            "metrics": {},
        },
    )
    model_entry["predictions"][split] = predictions
    model_entry["metrics"][split] = results
    if metadata:
        model_entry.update(metadata)
    save_model_results_to_json(model_results, output_path)
    return model_entry


def predict_and_store_log_return_model(
    model,
    model_name,
    train_inputs,
    test_inputs,
    train_targets,
    test_targets,
    train_current_prices,
    test_current_prices,
    target_scaler,
    model_results,
    output_path="model_results.json",
):
    """Predict a log-return model, reconstruct prices, and store both splits."""
    model.eval()
    with torch.no_grad():
        normalized_train_predictions = model(train_inputs)
        normalized_test_predictions = model(test_inputs)

    predicted_train_log_returns = denormalize_tensor(
        normalized_train_predictions,
        target_scaler,
    ).numpy().flatten()
    predicted_test_log_returns = denormalize_tensor(
        normalized_test_predictions,
        target_scaler,
    ).numpy().flatten()
    actual_train_log_returns = denormalize_tensor(
        train_targets,
        target_scaler,
    ).numpy().flatten()
    actual_test_log_returns = denormalize_tensor(
        test_targets,
        target_scaler,
    ).numpy().flatten()

    predicted_train_prices = train_current_prices * np.exp(predicted_train_log_returns)
    predicted_test_prices = test_current_prices * np.exp(predicted_test_log_returns)
    actual_train_prices = train_current_prices * np.exp(actual_train_log_returns)
    actual_test_prices = test_current_prices * np.exp(actual_test_log_returns)

    train_metrics = evaluate_predictions(predicted_train_prices, actual_train_prices)
    test_metrics = evaluate_predictions(predicted_test_prices, actual_test_prices)
    store_model_prediction(
        model_results,
        model_name,
        predicted_train_prices,
        train_metrics,
        "train",
        output_path,
    )
    store_model_prediction(
        model_results,
        model_name,
        predicted_test_prices,
        test_metrics,
        "test",
        output_path,
    )

    return {
        "train_predictions": predicted_train_prices,
        "test_predictions": predicted_test_prices,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_actual_prices": actual_train_prices,
        "test_actual_prices": actual_test_prices,
    }


class TimeSeriesWindowDataset(Dataset):
    """PyTorch Dataset for sliding-window time-series data with configurable window size, offset, stride, and target horizon.

    Parameters
    ----------
    features : array-like or torch.Tensor of shape (N, D_in)
        Input feature time series.
    targets : array-like or torch.Tensor of shape (N,) or (N, D_out)
        Target time series (or subset of features).
    window_size : int, default=30
        Number of historical time steps in each input window (lookback sequence length L).
    offset : int, default=0
        Offset / lead time between the end of the input window and the target.
        - offset=0: Target begins immediately at index `start_idx + window_size` (1-step ahead / next immediate step).
        - offset=k (k > 0): Target begins at index `start_idx + window_size + k` (k extra steps ahead).
        - offset=-1: Target aligns with the last step of the input window (index `start_idx + window_size - 1`).
    target_size : int, default=1
        Number of future steps in the target window (for multi-step forecasting).
    stride : int, default=1
        Step size between consecutive sliding window sample start positions.
    valid_indices : list or array of int, optional
        Explicit sample start indices (useful when creating train/validation/test splits from a single continuous array).
    dates : array-like or pd.DatetimeIndex, optional
        Timestamps corresponding to rows in features/targets for metadata retrieval.
    device : torch.device or str, optional
        Target device for tensors.
    dtype : torch.dtype, default=torch.float32
        Data type for output tensors.
    """

    def __init__(
        self,
        features,
        targets,
        window_size=30,
        offset=0,
        target_size=1,
        stride=1,
        valid_indices=None,
        dates=None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.features = _to_numpy(features)
        self.targets = _to_numpy(targets)
        self.window_size = int(window_size)
        self.offset = int(offset)
        self.target_size = int(target_size)
        self.stride = int(stride)
        self.device = device
        self.dtype = dtype

        if len(self.features) != len(self.targets):
            raise ValueError(
                f"Features length ({len(self.features)}) and targets length ({len(self.targets)}) must match."
            )

        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.target_size <= 0:
            raise ValueError("target_size must be a positive integer.")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.total_len = len(self.features)
        self.dates = np.asarray(dates) if dates is not None else None
        if self.dates is not None and len(self.dates) != self.total_len:
            raise ValueError("Dates array length must match features length.")

        if valid_indices is not None:
            self.indices = np.asarray(valid_indices, dtype=np.int64)
        else:
            indices = []
            for start_idx in range(0, self.total_len, self.stride):
                target_start = start_idx + self.window_size + self.offset
                target_end = target_start + self.target_size
                if start_idx + self.window_size <= self.total_len and 0 <= target_start and target_end <= self.total_len:
                    indices.append(start_idx)
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        x_window = self.features[start_idx:end_idx]

        target_start = start_idx + self.window_size + self.offset
        target_end = target_start + self.target_size
        y_window = self.targets[target_start:target_end]

        x_tensor = torch.as_tensor(x_window, dtype=self.dtype, device=self.device)
        y_tensor = torch.as_tensor(y_window, dtype=self.dtype, device=self.device)

        if self.targets.shape[1] == 1:
            y_tensor = y_tensor.squeeze(-1)
        if self.target_size == 1 and y_tensor.ndim == 1 and y_tensor.numel() == 1:
            y_tensor = y_tensor.squeeze(0)

        return x_tensor, y_tensor

    def get_target_indices(self):
        """Return the target starting indices for all samples in the dataset."""
        return self.indices + self.window_size + self.offset

    def get_target_dates(self):
        """Return target dates for all samples in the dataset (if dates were provided)."""
        if self.dates is None:
            return None
        target_indices = self.get_target_indices()
        return self.dates[target_indices]

    def get_input_window_dates(self, idx):
        """Return the date range for the input window of sample idx."""
        if self.dates is None:
            return None
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        return self.dates[start_idx:end_idx]


def create_time_series_dataloaders(
    data,
    target_column="INR",
    feature_columns=None,
    window_size=30,
    offset=0,
    target_size=1,
    stride=1,
    batch_size=64,
    train_fraction=0.8,
    validation_fraction=0.1,
    scale_features=True,
    scale_targets=True,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    device=None,
    dtype=torch.float32,
):
    """Create train, validation, and test TimeSeriesWindowDataset and DataLoader objects from tabular time series.

    Parameters
    ----------
    data : pd.DataFrame
        Time series dataframe sorted in chronological order.
    target_column : str or list of str, default="INR"
        Column name(s) for the prediction target.
    feature_columns : list of str, optional
        Column names to use as input features. Defaults to all columns except target_column if not specified.
    window_size : int, default=30
        Lookback sequence length (number of time steps per input sample).
    offset : int, default=0
        Offset from the end of the input window to the start of the target.
        - offset=0: Target is the immediate next time step (index t + window_size).
        - offset=k: Target is k additional steps ahead (index t + window_size + k).
    target_size : int, default=1
        Number of consecutive target steps to predict.
    stride : int, default=1
        Stride / step size between consecutive window starts in training set.
    batch_size : int, default=64
        Batch size for the DataLoader.
    train_fraction : float, default=0.8
        Fraction of data for training.
    validation_fraction : float, default=0.1
        Fraction of data for validation. Remaining fraction is used for testing.
    scale_features : bool, default=True
        Whether to fit a StandardScaler on train features and scale the full sequence.
    scale_targets : bool, default=True
        Whether to fit a StandardScaler on train targets and scale the full sequence.
    shuffle_train : bool, default=True
        Whether to shuffle training batches.
    num_workers : int, default=0
        Number of subprocesses for data loading.
    pin_memory : bool, default=False
        Whether DataLoader should pin memory for faster GPU transfers.
    device : torch.device or str, optional
        Target device for dataset tensors.
    dtype : torch.dtype, default=torch.float32
        Data type for tensors.

    Returns
    -------
    dict
        Dictionary containing:
        - "train_loader", "val_loader", "test_loader"
        - "train_dataset", "val_dataset", "test_dataset"
        - "feature_scaler", "target_scaler"
        - "feature_columns", "target_column"
        - "dates", "window_size", "offset", "target_size"
    """
    if isinstance(data, pd.DataFrame):
        df = data.sort_index().copy()
        dates = df.index.to_numpy()
    else:
        raise TypeError("data must be a pandas DataFrame.")

    target_cols = [target_column] if isinstance(target_column, str) else list(target_column)
    if feature_columns is None:
        feature_cols = [col for col in df.columns if col not in target_cols]
    else:
        feature_cols = list(feature_columns)

    raw_features = df[feature_cols].to_numpy(dtype=np.float32)
    raw_targets = df[target_cols].to_numpy(dtype=np.float32)
    total_len = len(df)

    train_cutoff = int(total_len * train_fraction)
    val_cutoff = train_cutoff + int(total_len * validation_fraction)

    if train_cutoff <= window_size + offset:
        raise ValueError(
            f"Training split ({train_cutoff} rows) is too small for window_size={window_size} and offset={offset}."
        )

    # Fit scalers strictly on training portion
    if scale_features:
        feature_scaler = StandardScaler().fit(raw_features[:train_cutoff])
        scaled_features = feature_scaler.transform(raw_features).astype(np.float32)
    else:
        feature_scaler = None
        scaled_features = raw_features

    if scale_targets:
        target_scaler = StandardScaler().fit(raw_targets[:train_cutoff])
        scaled_targets = target_scaler.transform(raw_targets).astype(np.float32)
    else:
        target_scaler = None
        scaled_targets = raw_targets

    # Determine sample start indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for start_idx in range(0, total_len):
        target_start = start_idx + window_size + offset
        target_end = target_start + target_size

        if start_idx + window_size <= total_len and 0 <= target_start and target_end <= total_len:
            if target_end <= train_cutoff:
                if start_idx % stride == 0:
                    train_indices.append(start_idx)
            elif train_cutoff <= target_start < val_cutoff and target_end <= val_cutoff:
                val_indices.append(start_idx)
            elif target_start >= val_cutoff:
                test_indices.append(start_idx)

    train_dataset = TimeSeriesWindowDataset(
        features=scaled_features,
        targets=scaled_targets,
        window_size=window_size,
        offset=offset,
        target_size=target_size,
        stride=stride,
        valid_indices=train_indices,
        dates=dates,
        device=device,
        dtype=dtype,
    )

    val_dataset = TimeSeriesWindowDataset(
        features=scaled_features,
        targets=scaled_targets,
        window_size=window_size,
        offset=offset,
        target_size=target_size,
        stride=1,
        valid_indices=val_indices,
        dates=dates,
        device=device,
        dtype=dtype,
    )

    test_dataset = TimeSeriesWindowDataset(
        features=scaled_features,
        targets=scaled_targets,
        window_size=window_size,
        offset=offset,
        target_size=target_size,
        stride=1,
        valid_indices=test_indices,
        dates=dates,
        device=device,
        dtype=dtype,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if len(train_dataset) > 0 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if len(test_dataset) > 0 else None

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_columns": feature_cols,
        "target_column": target_cols[0] if len(target_cols) == 1 else target_cols,
        "raw_features": raw_features,
        "raw_targets": raw_targets,
        "scaled_features": scaled_features,
        "scaled_targets": scaled_targets,
        "dates": dates,
        "window_size": window_size,
        "offset": offset,
        "target_size": target_size,
        "stride": stride,
    }


def train_torch_model_dataloader(
    model,
    optimizer,
    loss_function,
    train_loader,
    val_loader=None,
    epochs=100,
    device=None,
    log_every=10,
    scheduler=None,
    early_stopping_patience=None,
):
    """Train a PyTorch model using DataLoaders with validation and optional early stopping."""
    target_device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
    model.to(target_device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            if predictions.shape != batch_y.shape:
                if predictions.ndim == 2 and predictions.shape[1] == 1 and batch_y.ndim == 1:
                    predictions = predictions.squeeze(1)
                elif batch_y.ndim == 2 and batch_y.shape[1] == 1 and predictions.ndim == 1:
                    predictions = predictions.unsqueeze(1)

            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        epoch_train_loss = running_train_loss / max(num_train_batches, 1)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            running_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_x = val_x.to(target_device)
                    val_y = val_y.to(target_device)
                    val_preds = model(val_x)
                    if val_preds.shape != val_y.shape:
                        if val_preds.ndim == 2 and val_preds.shape[1] == 1 and val_y.ndim == 1:
                            val_preds = val_preds.squeeze(1)
                        elif val_y.ndim == 2 and val_y.shape[1] == 1 and val_preds.ndim == 1:
                            val_preds = val_preds.unsqueeze(1)
                    v_loss = loss_function(val_preds, val_y)
                    running_val_loss += v_loss.item()
                    num_val_batches += 1

            epoch_val_loss = running_val_loss / max(num_val_batches, 1)
            val_losses.append(epoch_val_loss)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            if early_stopping_patience is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if log_every:
                            print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights (Val Loss: {best_val_loss:.4f}).")
                        if best_state is not None:
                            model.load_state_dict(best_state)
                        break
        elif scheduler is not None:
            scheduler.step()

        if log_every and (epoch + 1) % log_every == 0:
            val_str = f", Val Loss: {epoch_val_loss:.4f}" if epoch_val_loss is not None else ""
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_train_loss:.4f}{val_str}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss if val_loader is not None else None,
    }


def evaluate_torch_model_dataloader(
    model,
    dataloader,
    device=None,
    target_scaler=None,
    previous_values=None,
    dates=None,
    title="Model Predictions vs Targets",
    plot=False,
):
    """Evaluate a PyTorch model on a DataLoader, returning predictions, actual targets, and metrics."""
    target_device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
    model.to(target_device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(target_device)
            preds = model(batch_x)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    if target_scaler is not None:
        orig_shape = preds_arr.shape
        preds_2d = preds_arr.reshape(-1, 1) if preds_arr.ndim == 1 else preds_arr
        targets_2d = targets_arr.reshape(-1, 1) if targets_arr.ndim == 1 else targets_arr

        preds_arr = target_scaler.inverse_transform(preds_2d)
        targets_arr = target_scaler.inverse_transform(targets_2d)

        if len(orig_shape) == 1 or (len(orig_shape) == 2 and orig_shape[1] == 1):
            preds_arr = preds_arr.flatten()
            targets_arr = targets_arr.flatten()

    target_dates = dates
    if target_dates is None and hasattr(dataloader.dataset, "get_target_dates"):
        target_dates = dataloader.dataset.get_target_dates()

    metrics = calculate_directional_metrics(preds_arr, targets_arr, previous_values)

    if plot:
        plot_predictions_vs_targets(
            preds_arr,
            targets_arr,
            x_values=target_dates,
            title=title,
        )

    return {
        "predictions": preds_arr,
        "targets": targets_arr,
        "metrics": metrics,
        "dates": target_dates,
    }


def predict_and_store_window_model(
    model,
    model_name,
    loaders_dict,
    model_results,
    device=None,
    output_path="model_results.json",
    metadata=None,
):
    """Evaluate train, validation, and test dataloaders for a windowed model, storing predictions and metrics."""
    target_scaler = loaders_dict.get("target_scaler")
    train_loader = loaders_dict.get("train_loader")
    val_loader = loaders_dict.get("val_loader")
    test_loader = loaders_dict.get("test_loader")

    train_eval = None
    if train_loader is not None:
        train_eval = evaluate_torch_model_dataloader(
            model,
            train_loader,
            device=device,
            target_scaler=target_scaler,
        )
        store_model_prediction(
            model_results,
            model_name,
            train_eval["predictions"],
            train_eval["metrics"],
            "train",
            output_path=output_path,
            metadata=metadata,
        )

    val_eval = None
    if val_loader is not None:
        val_eval = evaluate_torch_model_dataloader(
            model,
            val_loader,
            device=device,
            target_scaler=target_scaler,
        )
        store_model_prediction(
            model_results,
            model_name,
            val_eval["predictions"],
            val_eval["metrics"],
            "validation",
            output_path=output_path,
        )

    test_eval = None
    if test_loader is not None:
        test_eval = evaluate_torch_model_dataloader(
            model,
            test_loader,
            device=device,
            target_scaler=target_scaler,
        )
        store_model_prediction(
            model_results,
            model_name,
            test_eval["predictions"],
            test_eval["metrics"],
            "test",
            output_path=output_path,
        )

    return {
        "train": train_eval,
        "validation": val_eval,
        "test": test_eval,
    }
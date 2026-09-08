import json
from itertools import product
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    result["EMA_5"] = series.ewm(span=5, adjust=False).mean()
    result["EMA_10"] = series.ewm(span=10, adjust=False).mean()
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    result["RSI"] = 100 - (100 / (1 + gain / loss))
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
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from helpers.data_loaders import StockWindowDataset, load_processed_stock
from helpers.utils import (
    calculate_directional_metrics,
    train_torch_model_dataloader,
)
from loss_functions import BigMoveAwareLoss
from models import LSTMForecaster
from rl_model import Action, QLearningAgent, TradingEnvironment, forecast_batches


if __name__ == "__main__":
    STOCK = "LTIM"
    FEATURES = (
        "returns", "log_returns", "rsi_14", "trend_slope_pct_20",
        "rolling_volatility_30", "relative_volume_20", "log_volume",
    )
    LOOKBACK = 30
    BATCH_SIZE = 128
    EPOCHS = 20
    FOLD_EPOCHS = 5
    RL_EPOCHS = 10
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep full sessions together and build windows separately within each split.
    data = load_processed_stock(STOCK)
    row_dates = data.index.normalize()
    sessions = row_dates.unique().sort_values()
    train_end = int(0.8 * len(sessions))
    val_end = int(0.9 * len(sessions))
    if not 0 < train_end < val_end < len(sessions):
        raise ValueError(f"{STOCK} has too few sessions for an 80/10/10 split.")

    splits = {
        "train": data.loc[row_dates < sessions[train_end]],
        "validation": data.loc[(row_dates >= sessions[train_end]) & (row_dates < sessions[val_end])],
        "test": data.loc[row_dates >= sessions[val_end]],
    }
    loaders = {}
    for name, frame in splits.items():
        dataset = StockWindowDataset(frame, FEATURES, lookback=LOOKBACK, horizon=1)
        if not len(dataset):
            raise ValueError(f"{STOCK} has no eligible windows in the {name} split.")
        loaders[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=name == "train")
        print(f"{name}: {frame.index[0].date()} to {frame.index[-1].date()}, {len(dataset):,} windows")

    loss_function = BigMoveAwareLoss(sigma_threshold=0.005)

    # Fit expanding LSTM folds and forecast only later training sessions.
    train_dataset = loaders["train"].dataset
    train_dates = train_dataset.target_timestamps.normalize()
    fold_days = np.array_split(train_dates.unique().to_numpy(), 4)
    if any(not len(days) for days in fold_days):
        raise ValueError("Four nonempty training session blocks are required for RL forecasts.")
    policy_forecasts, policy_returns, policy_dates = [], [], []
    for fold, days in enumerate(fold_days[1:], start=1):
        fit_indices = np.flatnonzero(train_dates < days[0]).tolist()
        forecast_indices = np.flatnonzero(train_dates.isin(days)).tolist()
        fold_model = LSTMForecaster(input_size=len(FEATURES)).to(device)
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)
        train_torch_model_dataloader(
            fold_model, fold_optimizer, loss_function,
            DataLoader(Subset(train_dataset, fit_indices), batch_size=BATCH_SIZE, shuffle=True),
            epochs=FOLD_EPOCHS, device=device, log_every=0,
        )
        fold_predictions, fold_actual = forecast_batches(
            fold_model, DataLoader(Subset(train_dataset, forecast_indices), batch_size=BATCH_SIZE), device,
        )
        policy_forecasts.extend(fold_predictions.tolist())
        policy_returns.extend(fold_actual.tolist())
        policy_dates.extend(train_dataset.target_timestamps.take(forecast_indices))
        print(f"RL forecast fold {fold}: {len(fit_indices):,} fit, {len(forecast_indices):,} forecast windows")

    model = LSTMForecaster(input_size=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = train_torch_model_dataloader(
        model, optimizer, loss_function, loaders["train"], loaders["validation"],
        epochs=EPOCHS, device=device, log_every=1, early_stopping_patience=5,
    )
    print(f"Best validation loss: {history['best_val_loss']:.8f}")

    validation_predictions, validation_actual = forecast_batches(model, loaders["validation"], device)
    train_env = TradingEnvironment(policy_forecasts, policy_returns, policy_dates)
    validation_env = TradingEnvironment(
        validation_predictions.tolist(), validation_actual.tolist(),
        loaders["validation"].dataset.target_timestamps,
    )
    agent = QLearningAgent(np.quantile(policy_forecasts, [0.2, 0.4, 0.6, 0.8]))
    best_return = agent.run(validation_env)["net_return_pct"]
    best_q = [row.copy() for row in agent.q]
    for epoch in range(RL_EPOCHS):
        agent.run(train_env, learn=True, epsilon=max(0.05, 0.3 * 0.8 ** epoch))
        validation_return = agent.run(validation_env)["net_return_pct"]
        if validation_return > best_return:
            best_return = validation_return
            best_q = [row.copy() for row in agent.q]
    agent.q = best_q
    print(f"Best RL validation return: {best_return:+.2f}%")

    predictions, actual = forecast_batches(model, loaders["test"], device)
    print(f"Test loss: {loss_function(predictions, actual).item():.8f}")
    predictions = predictions.numpy()
    actual = actual.numpy()
    metrics = calculate_directional_metrics(predictions, actual, previous_values=np.zeros_like(actual))
    pprint({name: metrics[name] for name in (
        "mae", "mse", "rmse", "directional_accuracy", "up_recall", "down_recall",
    )})
    print(f"Always-up baseline: {100 * np.mean(actual >= 0):.2f}%")

    test_env = TradingEnvironment(
        predictions, actual, loaders["test"].dataset.target_timestamps,
    )
    print("RL policy:")
    pprint(agent.run(test_env))
    rl_trades = test_env.trades.copy()
    observation = test_env.reset()
    while observation is not None:
        forecast, long, _ = observation
        desired_long = forecast > 0.0001
        action = Action.HOLD if long == desired_long else (Action.BUY if desired_long else Action.SELL)
        observation, _, _ = test_env.step(action)
    print("LSTM threshold baseline:")
    pprint(test_env.summary())
    print("Cash baseline: 0.00% return, no fees or exposure.")

    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(exist_ok=True)
    trade_columns = ["entry_time", "exit_time", "held_minutes", "exit_reason",
                     "net_return_pct", "net_pnl_per_10000", "fees_per_10000"]
    for name, trades in (("rl_policy", rl_trades), ("lstm_threshold", test_env.trades)):
        path = report_dir / f"{STOCK}_{name}_trades.csv"
        report = pd.DataFrame(trades, columns=trade_columns, dtype=object)
        pnl = report["net_pnl_per_10000"].astype(float)
        report["cumulative_net_pnl_per_10000"] = pnl.cumsum()
        report["cumulative_fees_per_10000"] = report["fees_per_10000"].astype(float).cumsum()
        report["equity_per_10000"] = 10_000 + report["cumulative_net_pnl_per_10000"]
        report["cumulative_growth_factor"] = report["equity_per_10000"] / 10_000
        report["cumulative_net_return_pct"] = report["cumulative_net_pnl_per_10000"] / 100
        cumulative_losses = -pnl.clip(upper=0).cumsum()
        report["cumulative_profit_factor"] = pnl.clip(lower=0).cumsum() / cumulative_losses.where(cumulative_losses > 0)
        report.to_csv(path, index=False)
        print(f"Saved {len(trades)} closed trades to {path}")

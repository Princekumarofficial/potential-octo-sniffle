"""
General Time-Series Trainer & Trading Strategy Optimizer.

Universal pipeline for training directional LSTM forecasting models on any time-series
(Forex, Equities, Commodities, Crypto, Macro indices), performing backtesting with
Take-Profit, Stop-Loss, and Risk-Reward management, and finding optimal trading parameters
using a Genetic Algorithm.

All artifacts (metrics, predictions, trade logs, PyTorch models, and high-resolution plots)
are automatically saved in a structured output folder.

Usage as a Python Module:
-------------------------
>>> from general_trainer import GeneralTrainer
>>> trainer = GeneralTrainer(
...     data="FRED",                  # or "path/to/stock.csv", or pd.DataFrame
...     target_column="EUR",          # or "Close", "INR", etc.
...     output_dir="runs/EUR_run",
...     epochs=60,
...     ga_fitness_metric="composite"
... )
>>> results = trainer.train()

Usage from the Command Line:
----------------------------
$ python general_trainer.py --data FRED --target EUR --output_dir runs/EUR
$ python general_trainer.py --data FRED/sap500.csv --target Close --output_dir runs/SP500 --epochs 50
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    AdvancedProfitCalculator,
    calculate_directional_metrics,
    calculate_technical_indicators,
    create_time_series_dataloaders,
    evaluate_predictions,
    evaluate_torch_model_dataloader,
    load_fred_data,
    optimize_strategy_ga,
    prepare_data,
    train_torch_model_dataloader,
    train_on_val_data,
)


# ────────────────────────────────────────────────────────────────────────────
# Core PyTorch Modules
# ────────────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """LSTM-based time-series forecasting model with linear projection."""
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out.squeeze(-1)


class BigMoveAwareLoss(nn.Module):
    """
    Directional loss function that heavily penalizes sign mismatches (wrong direction)
    and penalizes missing large tail-risk market moves (moves >= sigma_threshold).
    """
    def __init__(self, dir_penalty=0.5, mag_penalty=2.0, sigma_threshold=1.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.dir_penalty = dir_penalty
        self.mag_penalty = mag_penalty
        self.sigma_threshold = sigma_threshold

    def forward(self, y_pred, y_true):
        sq_error = self.mse(y_pred, y_true)

        weight = 1.0 + self.mag_penalty * torch.abs(y_true)
        big_move_mask = (torch.abs(y_true) >= self.sigma_threshold).float()
        weight = weight + (big_move_mask * self.mag_penalty * 2.0)

        weighted_mse = sq_error * weight

        sign_mismatch = torch.relu(-torch.sign(y_pred) * torch.sign(y_true))
        dir_weight = torch.abs(y_true) + (big_move_mask * 2.0)
        dir_loss = sign_mismatch * dir_weight * self.dir_penalty

        return torch.mean(weighted_mse + dir_loss)


# ────────────────────────────────────────────────────────────────────────────
# Universal Time-Series Trainer Class
# ────────────────────────────────────────────────────────────────────────────
class GeneralTrainer:
    """
    Universal Time-Series Trainer and Quant Backtesting Pipeline.

    Parameters
    ----------
    data : str, Path, or pd.DataFrame
        Data source. Can be:
        - "FRED" or directory containing FRED files.
        - Path to a CSV file (e.g. stock / index / forex CSV).
        - A pandas DataFrame with a DatetimeIndex or a Date column.
    target_column : str, default="Close"
        Name of the target series / price column to forecast and trade.
    feature_columns : list of str, optional
        Exogenous feature columns. If None, automatically infers remaining numeric columns.
    date_column : str, optional
        Name of date column in CSV. If None, auto-detects from common names ('Date', 'datetime', etc.).
    output_dir : str or Path, default="runs"
        Parent directory where experiment results folder will be stored.
    run_name : str, optional
        Name of output subfolder. If None, auto-generates '{target_column}_{timestamp}'.
    window_size : int, default=10
        Lookback sequence length (number of historical days in each sample).
    hidden_size : int, default=32
        Number of hidden units in LSTM layer.
    num_layers : int, default=1
        Number of recurrent LSTM layers.
    dropout : float, default=0.0
        Dropout probability.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay (L2 regularization).
    epochs : int, default=80
        Maximum training epochs.
    batch_size : int, default=32
        Batch size for PyTorch DataLoader.
    patience : int, default=10
        Early stopping patience (epochs without validation loss improvement).
    train_fraction : float, default=0.8
        Fraction of data for training split.
    val_fraction : float, default=0.1
        Fraction of data for validation split. Remainder is used for testing.
    dir_penalty : float, default=0.5
        Directional sign mismatch penalty weight.
    mag_penalty : float, default=2.0
        Magnitude penalty weight for large market returns.
    sigma_threshold : float, default=1.5
        Standard deviation threshold to trigger tail-move penalty.
    ga_optimize : bool, default=True
        Whether to run Genetic Algorithm hyperparameter optimization on trading strategy.
    ga_fitness_metric : str, default="composite"
        Fitness metric for GA: 'composite', 'return', 'sharpe', 'sortino', 'calmar', 'profit_factor'.
    ga_population_size : int, default=40
        Candidate population size per GA generation.
    ga_generations : int, default=25
        Number of GA evolutionary cycles.
    initial_capital : float, default=10000.0
        Initial trading portfolio equity.
    fee_per_trade : float, default=0.0001
        Trading friction (broker commission + bid-ask spread) per trade.
    device : str or torch.device, optional
        Compute device. Automatically detects CUDA -> MPS -> CPU if None.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print status and progress logs.
    """

    def __init__(
        self,
        data,
        target_column="Close",
        feature_columns=None,
        date_column=None,
        output_dir="runs",
        run_name=None,
        window_size=10,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=80,
        batch_size=32,
        patience=10,
        train_fraction=0.8,
        val_fraction=0.1,
        train_on_val=False,
        val_epochs=15,
        val_lr=1e-4,
        dir_penalty=0.5,
        mag_penalty=2.0,
        sigma_threshold=1.5,
        ga_optimize=True,
        ga_fitness_metric="composite",
        ga_population_size=40,
        ga_generations=25,
        initial_capital=10000.0,
        fee_per_trade=0.0001,
        device=None,
        random_state=42,
        verbose=True,
    ):
        self.raw_data_input = data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.date_column = date_column
        self.output_dir = Path(output_dir)
        self.run_name = run_name or f"{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_folder = self.output_dir / self.run_name
        self.plots_folder = self.run_folder / "plots"

        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.train_on_val = train_on_val
        self.val_epochs = val_epochs
        self.val_lr = val_lr
        self.dir_penalty = dir_penalty
        self.mag_penalty = mag_penalty
        self.sigma_threshold = sigma_threshold

        self.ga_optimize = ga_optimize
        self.ga_fitness_metric = ga_fitness_metric
        self.ga_population_size = ga_population_size
        self.ga_generations = ga_generations
        self.initial_capital = initial_capital
        self.fee_per_trade = fee_per_trade
        self.random_state = random_state
        self.verbose = verbose

        # Determine compute device
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Container for intermediate and final results
        self.processed_df = None
        self.loaders = None
        self.model = None
        self.training_history = None
        self.evaluation_results = {}
        self.ga_results = None
        self.trading_results = {}
        self.final_metrics = {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ────────────────────────────────────────────────────────────────────────
    # 1. Load & Prepare Data
    # ────────────────────────────────────────────────────────────────────────
    def load_and_prepare_data(self):
        """Load, validate, parse dates, and compute technical indicators."""
        self._log(f"Loading and preparing data (Target: {self.target_column})...")

        # Case 1: "FRED" or directory containing FRED files
        if isinstance(self.raw_data_input, (str, Path)) and (
            str(self.raw_data_input).upper() == "FRED" or (Path(self.raw_data_input).is_dir())
        ):
            fred_dir = str(self.raw_data_input)
            df = prepare_data(load_fred_data(fred_dir))

        # Case 2: DataFrame passed directly
        elif isinstance(self.raw_data_input, pd.DataFrame):
            df = self.raw_data_input.copy()

        # Case 3: CSV file path
        elif isinstance(self.raw_data_input, (str, Path)):
            csv_path = Path(self.raw_data_input)
            if not csv_path.exists():
                raise FileNotFoundError(f"Data file not found: {csv_path}")

            # Peek headers
            temp_df = pd.read_csv(csv_path, nrows=5)
            date_col = self.date_column

            if date_col is None:
                for candidate in ["Date", "date", "Timestamp", "timestamp", "datetime", "Time", "observation_date"]:
                    if candidate in temp_df.columns:
                        date_col = candidate
                        break

            if date_col is not None:
                df = pd.read_csv(csv_path, parse_dates=[date_col], index_col=date_col)
            else:
                df = pd.read_csv(csv_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        pass
        else:
            raise TypeError(f"Unsupported data input type: {type(self.raw_data_input)}")

        # Ensure index is sorted chronologically
        df = df.sort_index()

        # Clean numeric columns (remove string formatting like commas if present)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(str).str.replace(",", "").astype(float)
                except Exception:
                    pass

        # Validate target column
        if self.target_column not in df.columns:
            available = list(df.columns)
            raise ValueError(
                f"Target column '{self.target_column}' not found in data columns: {available}"
            )

        # Drop rows with missing target price
        df = df.dropna(subset=[self.target_column]).copy()

        # Compute complete suite of Technical Indicators
        self._log(f"Calculating technical indicators for {self.target_column}...")
        ta_df = calculate_technical_indicators(df, target_column=self.target_column)

        # 1-step ahead Target: tomorrow's return
        ta_df["TARGET_LOG_RETURN_NEXT"] = ta_df["Log_Returns"].shift(-1)
        ta_df = ta_df.dropna().copy()

        # Determine feature columns
        ta_feature_names = [
            "MA_5", "MA_10", "MA_30", "MA_50", "MA_100",
            "EMA_5", "EMA_10", "EMA_30", "EMA_50", "EMA_100",
            "RSI",
            "BB_Middle", "BB_Upper", "BB_Lower",
            "Volatility",
            "ATR_14_proxy", "ATR_%_proxy",
            "Log_Returns",
        ]

        if self.feature_columns is not None:
            exog_cols = [c for c in self.feature_columns if c in df.columns and c != self.target_column]
        else:
            # Auto-include any other numeric series present in original dataframe
            exog_cols = [
                c for c in df.columns
                if c != self.target_column and pd.api.types.is_numeric_dtype(df[c])
            ]

        self.full_feature_cols = [self.target_column, *exog_cols, *ta_feature_names]
        # Remove any duplicates while preserving order
        self.full_feature_cols = list(dict.fromkeys(self.full_feature_cols))

        self.processed_df = ta_df
        self._log(
            f"Data prepared: {len(ta_df)} rows, {len(self.full_feature_cols)} features. "
            f"Date range: {ta_df.index[0]} to {ta_df.index[-1]}"
        )
        return self.processed_df

    # ────────────────────────────────────────────────────────────────────────
    # 2. Build Windowed DataLoaders
    # ────────────────────────────────────────────────────────────────────────
    def create_dataloaders(self):
        """Construct sliding window datasets and train/val/test PyTorch DataLoaders."""
        if self.processed_df is None:
            self.load_and_prepare_data()

        self._log("Creating sliding-window time-series DataLoaders...")
        self.loaders = create_time_series_dataloaders(
            data=self.processed_df,
            target_column="TARGET_LOG_RETURN_NEXT",
            feature_columns=self.full_feature_cols,
            window_size=self.window_size,
            offset=0,
            target_size=1,
            stride=1,
            batch_size=self.batch_size,
            train_fraction=self.train_fraction,
            validation_fraction=self.val_fraction,
            scale_features=True,
            scale_targets=True,
            shuffle_train=True,
            device=self.device,
        )

        n_train = len(self.loaders["train_dataset"])
        n_val = len(self.loaders["val_dataset"])
        n_test = len(self.loaders["test_dataset"])
        self._log(f"Dataset splits: Train={n_train}, Validation={n_val}, Test={n_test}")
        return self.loaders

    # ────────────────────────────────────────────────────────────────────────
    # 3. Model Training
    # ────────────────────────────────────────────────────────────────────────
    def train_model(self):
        """Initialize LSTM and train with BigMoveAwareLoss, LR scheduler, and early stopping."""
        if self.loaders is None:
            self.create_dataloaders()

        input_size = len(self.full_feature_cols)
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=1,
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = BigMoveAwareLoss(
            dir_penalty=self.dir_penalty,
            mag_penalty=self.mag_penalty,
            sigma_threshold=self.sigma_threshold,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )

        self._log(f"Training LSTM model on {self.device} (Max Epochs: {self.epochs}, Patience: {self.patience})...")
        self.training_history = train_torch_model_dataloader(
            model=self.model,
            optimizer=optimizer,
            loss_function=loss_fn,
            train_loader=self.loaders["train_loader"],
            val_loader=self.loaders["val_loader"],
            epochs=self.epochs,
            device=self.device,
            log_every=10 if self.verbose else 0,
            scheduler=scheduler,
            early_stopping_patience=self.patience,
        )
        self._log("Model training completed. Best validation weights restored.")

        # Stage 2 (Optional): Train / fine-tune on validation data
        if self.train_on_val and self.val_epochs > 0 and self.loaders.get("val_loader"):
            self._log(f"\n--- Fine-tuning model on Validation Data ({self.val_epochs} epochs, lr={self.val_lr}) ---")
            val_history = train_on_val_data(
                model=self.model,
                val_loader=self.loaders["val_loader"],
                loss_function=loss_fn,
                epochs=self.val_epochs,
                lr=self.val_lr,
                device=self.device,
                log_every=max(1, self.val_epochs // 3) if self.verbose else 0,
                shuffle=True,
            )
            self.training_history["val_train_losses"] = val_history.get("val_train_losses", [])

        return self.model, self.training_history

    # ────────────────────────────────────────────────────────────────────────
    # 4. Model Evaluation & Price Reconstruction
    # ────────────────────────────────────────────────────────────────────────
    def evaluate(self):
        """Evaluate model across all splits, compute metrics, and reconstruct price forecasts."""
        if self.model is None:
            self.train_model()

        self._log("Evaluating model across Train, Validation, and Test splits...")
        target_scaler = self.loaders.get("target_scaler")
        splits = {
            "train": self.loaders["train_loader"],
            "validation": self.loaders["val_loader"],
            "test": self.loaders["test_loader"],
        }

        eval_summary = {}

        for split_name, loader in splits.items():
            if loader is None:
                continue

            eval_res = evaluate_torch_model_dataloader(
                model=self.model,
                dataloader=loader,
                device=self.device,
                target_scaler=target_scaler,
                plot=False,
            )

            # Dates identify source row whose price is the base for next return
            dates = eval_res["dates"]
            current_prices = self.processed_df.loc[dates, self.target_column].to_numpy().reshape(-1)
            pred_returns = np.asarray(eval_res["predictions"]).reshape(-1)
            actual_returns = np.asarray(eval_res["targets"]).reshape(-1)

            pred_prices = current_prices * np.exp(pred_returns)
            actual_prices = current_prices * np.exp(actual_returns)

            price_metrics = evaluate_predictions(
                pred_prices,
                actual_prices,
                previous_values=current_prices,
            )

            self.evaluation_results[split_name] = {
                "dates": dates,
                "current_prices": current_prices,
                "pred_returns": pred_returns,
                "actual_returns": actual_returns,
                "pred_prices": pred_prices,
                "actual_prices": actual_prices,
                "metrics": eval_res["metrics"],
                "price_metrics": price_metrics,
            }

            m = eval_res["metrics"]
            self._log(
                f"{split_name.capitalize():>12s} Split — "
                f"MAE: {m['mae']:.5f} | RMSE: {m['rmse']:.5f} | "
                f"Directional Accuracy: {m['directional_accuracy']:.2f}%"
            )

            eval_summary[split_name] = {
                "return_mae": m["mae"],
                "return_rmse": m["rmse"],
                "directional_accuracy_pct": m["directional_accuracy"],
                "price_mae": price_metrics["mae"],
                "price_rmse": price_metrics["rmse"],
            }

        self.final_metrics["forecasting"] = eval_summary
        return self.evaluation_results

    # ────────────────────────────────────────────────────────────────────────
    # 5. Trading Backtest & Genetic Algorithm Optimization
    # ────────────────────────────────────────────────────────────────────────
    def run_trading_and_ga(self):
        """Run quantitative backtest and Genetic Algorithm strategy optimization."""
        if not self.evaluation_results:
            self.evaluate()

        val_res = self.evaluation_results.get("validation")
        test_res = self.evaluation_results.get("test")

        if val_res is None or test_res is None:
            self._log("Validation or test split unavailable for trading simulation.")
            return None

        # 1. Baseline Backtest (Default 1:2 RRR, 0.4% SL, 0.8% TP)
        self._log("\nRunning baseline trading backtest on Validation split...")
        val_baseline_calc = AdvancedProfitCalculator(
            predictions=val_res["pred_returns"],
            actual_returns=val_res["actual_returns"],
            initial_capital=self.initial_capital,
            stop_loss=0.004,
            risk_reward_ratio=2.0,
            threshold=0.0001,
            fee_per_trade=self.fee_per_trade,
            risk_per_trade=0.02,
            max_leverage=3.0,
            dates=val_res["dates"],
        )
        self.trading_results["validation_baseline"] = val_baseline_calc

        # 2. Genetic Algorithm Optimization on Validation Split
        if self.ga_optimize:
            self._log(
                f"\nExecuting Genetic Algorithm Optimization on Validation set "
                f"(Objective: {self.ga_fitness_metric.upper()}, Pop: {self.ga_population_size}, Gen: {self.ga_generations})..."
            )
            self.ga_results = optimize_strategy_ga(
                predictions=val_res["pred_returns"],
                actual_returns=val_res["actual_returns"],
                fitness_metric=self.ga_fitness_metric,
                population_size=self.ga_population_size,
                generations=self.ga_generations,
                fee_per_trade=self.fee_per_trade,
                random_state=self.random_state,
                dates=val_res["dates"],
                verbose=self.verbose,
            )

            best_params = self.ga_results["best_params"]
            self.trading_results["validation_ga"] = self.ga_results["best_calculator"]

            # 3. Test Set Out-of-Sample Verification
            self._log("\nEvaluating GA-evolved optimal strategy on Out-of-Sample Test set...")
            test_ga_calc = AdvancedProfitCalculator(
                predictions=test_res["pred_returns"],
                actual_returns=test_res["actual_returns"],
                initial_capital=self.initial_capital,
                stop_loss=best_params["stop_loss"],
                take_profit=best_params.get("take_profit"),
                risk_reward_ratio=best_params.get("risk_reward_ratio", 2.0),
                threshold=best_params.get("threshold", 0.0),
                risk_per_trade=best_params.get("risk_per_trade", 0.02),
                fee_per_trade=self.fee_per_trade,
                max_leverage=best_params.get("max_leverage", 3.0),
                dates=test_res["dates"],
            )
            self.trading_results["test_ga"] = test_ga_calc

            if self.verbose:
                print("\n" + "=" * 80)
                print("         OUT-OF-SAMPLE TEST SET PERFORMANCE (GA-OPTIMIZED)")
                print("=" * 80)
                test_ga_calc.summary()
        else:
            # If GA disabled, evaluate baseline on test
            test_calc = AdvancedProfitCalculator(
                predictions=test_res["pred_returns"],
                actual_returns=test_res["actual_returns"],
                initial_capital=self.initial_capital,
                stop_loss=0.004,
                risk_reward_ratio=2.0,
                threshold=0.0001,
                fee_per_trade=self.fee_per_trade,
                risk_per_trade=0.02,
                dates=test_res["dates"],
            )
            self.trading_results["test_baseline"] = test_calc

        return self.trading_results

    # ────────────────────────────────────────────────────────────────────────
    # 6. Save Artifacts, Plots & Summary Report
    # ────────────────────────────────────────────────────────────────────────
    def save_results(self):
        """Save all plots, predictions CSVs, trade logs, PyTorch model, and metrics."""
        self._log(f"\nSaving experiment artifacts to: {self.run_folder.resolve()}")
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        # 1. Save PyTorch Model Checkpoint
        model_path = self.run_folder / "model_best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": len(self.full_feature_cols),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "feature_columns": self.full_feature_cols,
                "target_column": self.target_column,
                "window_size": self.window_size,
            },
            model_path,
        )

        # 2. Save Predictions to CSV files
        for split_name, res in self.evaluation_results.items():
            pred_df = pd.DataFrame(
                {
                    "Date": res["dates"],
                    "Current_Price": res["current_prices"],
                    "Actual_Return": res["actual_returns"],
                    "Predicted_Return": res["pred_returns"],
                    "Actual_Price": res["actual_prices"],
                    "Predicted_Price": res["pred_prices"],
                    "Error": res["pred_returns"] - res["actual_returns"],
                }
            )
            csv_path = self.run_folder / f"predictions_{split_name}.csv"
            pred_df.to_csv(csv_path, index=False)

        # 3. Save Detailed Trade Logs
        active_calc = self.trading_results.get("test_ga") or self.trading_results.get("test_baseline")
        if active_calc is not None:
            trades_df = pd.DataFrame(active_calc.trades)
            trades_csv = self.run_folder / "trading_trades_test.csv"
            trades_df.to_csv(trades_csv, index=False)

        # 4. Save Plots
        self._generate_and_save_plots()

        # 5. Save Metrics JSON & Text Summary
        self._save_metrics_and_summary()
        self._log(f"All artifacts successfully written to: {self.run_folder}")

    def _generate_and_save_plots(self):
        """Render and save all diagnostic and performance plots."""
        # Plot 1: Training Loss Curve
        if self.training_history:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.training_history["train_losses"], label="Train Loss", color="#1E88E5", lw=1.8)
            if self.training_history.get("val_losses"):
                ax.plot(self.training_history["val_losses"], label="Validation Loss", color="#FF8F00", lw=1.8)
            ax.set_title(f"Model Training Loss Curve ({self.target_column})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("BigMoveAware Loss", fontsize=11)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout()
            fig.savefig(self.plots_folder / "1_training_loss.png", dpi=200)
            plt.close(fig)

        # Plot 2: Test Set Log Returns Predictions & Errors
        test_res = self.evaluation_results.get("test")
        if test_res:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(test_res["dates"], test_res["actual_returns"], label="Actual Log Return", color="#1E88E5", alpha=0.8)
            axes[0].plot(test_res["dates"], test_res["pred_returns"], label="Predicted Log Return", color="#FF8F00", linestyle="--", lw=1.5)
            axes[0].set_title(f"Test Set: Predicted vs Actual Log Returns ({self.target_column})", fontsize=12, fontweight="bold")
            axes[0].legend()
            axes[0].grid(True, linestyle=":", alpha=0.6)

            errors = test_res["pred_returns"] - test_res["actual_returns"]
            axes[1].plot(test_res["dates"], errors, color="#8E24AA", lw=1.0)
            axes[1].fill_between(test_res["dates"], 0, errors, color="#8E24AA", alpha=0.25)
            axes[1].axhline(0, color="black", lw=0.8)
            axes[1].set_title("Prediction Error Residuals", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Date", fontsize=11)
            axes[1].grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout()
            fig.savefig(self.plots_folder / "2_test_returns_predictions.png", dpi=200)
            plt.close(fig)

            # Plot 3: Actual vs Forecasted Reconstructed Price
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(test_res["dates"], test_res["actual_prices"], label="Actual Price", color="#0D47A1", lw=1.8)
            ax.plot(test_res["dates"], test_res["pred_prices"], label="Forecasted Price", color="#E65100", linestyle="--", lw=1.4)
            ax.set_title(f"Out-of-Sample Price Forecast ({self.target_column})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel(f"{self.target_column} Price", fontsize=11)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout()
            fig.savefig(self.plots_folder / "3_test_price_forecast.png", dpi=200)
            plt.close(fig)

        # Plot 4: GA Convergence
        if self.ga_results and "optimizer" in self.ga_results:
            try:
                hist_df = self.ga_results["history"]
                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                axes[0].plot(hist_df["generation"], hist_df["best_fitness"], label="Best Fitness", color="#1E88E5", lw=2.2)
                axes[0].plot(hist_df["generation"], hist_df["mean_fitness"], label="Mean Fitness", color="#FF8F00", linestyle="--", lw=1.5)
                axes[0].set_title(f"Genetic Algorithm Fitness Progression ({self.ga_fitness_metric.upper()})", fontsize=12, fontweight="bold")
                axes[0].set_ylabel("Fitness Score", fontsize=11)
                axes[0].legend()
                axes[0].grid(True, linestyle=":", alpha=0.6)

                opt = self.ga_results["optimizer"]
                colors = ["#43A047", "#8E24AA", "#00ACC1", "#E53935", "#3949AB"]
                for i, key in enumerate(opt.gene_keys):
                    vals = [h["params"][key] for h in opt.history]
                    low, high = opt.param_bounds[key]
                    norm_vals = [(v - low) / (high - low + 1e-9) for v in vals]
                    axes[1].plot(hist_df["generation"], norm_vals, label=f"{key} ({vals[-1]:.4f})", color=colors[i % len(colors)], lw=1.8)

                axes[1].set_title("Normalized Parameter Evolution [0, 1]", fontsize=11, fontweight="bold")
                axes[1].set_xlabel("Generation", fontsize=11)
                axes[1].set_ylabel("Normalized Value", fontsize=10)
                axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9)
                axes[1].grid(True, linestyle=":", alpha=0.6)
                fig.tight_layout()
                fig.savefig(self.plots_folder / "4_ga_convergence.png", dpi=200)
                plt.close(fig)
            except Exception as e:
                self._log(f"Warning: Failed to render GA plot: {e}")

        # Plot 5 & 6: Trading Dashboards (Val & Test)
        for split_label, key in [("val", "validation_ga"), ("val_base", "validation_baseline"), ("test", "test_ga")]:
            calc = self.trading_results.get(key)
            if calc is None:
                continue

            try:
                # Custom render saving to file
                m = calc.metrics
                fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2.2, 1.2, 1.4]})
                x_axis = calc.dates if calc.dates is not None else np.arange(len(calc.equity_curve))

                # Align lengths if date index was 1 element longer
                eq = calc.equity_curve
                bench = calc.benchmark_curve
                dd = m["drawdowns"]
                if calc.dates is not None and len(calc.dates) == len(eq) - 1:
                    eq = eq[1:]
                    bench = bench[1:]
                    dd = dd[1:]

                # Equity Curve
                axes[0].plot(x_axis, eq, label=f"Strategy Equity (${m['final_equity']:,.0f} / {m['strategy_return_pct']:+.1f}%)", color="#1E88E5", lw=2.2)
                axes[0].plot(x_axis, bench, label=f"Buy & Hold (${bench[-1]:,.0f} / {m['benchmark_return_pct']:+.1f}%)", color="#78909C", linestyle="--", lw=1.5)
                axes[0].axhline(y=calc.initial_capital, color="#B0BEC5", linestyle=":", lw=1.2, label=f"Initial Capital (${calc.initial_capital:,.0f})")
                axes[0].set_title(f"Cumulative Portfolio Value [{split_label.upper()}] (Alpha: {m['alpha_pct']:+.1f}% | Profit Factor: {m['profit_factor']:.2f} | Sharpe: {m['sharpe_ratio']:.2f})", fontsize=12, fontweight="bold")
                axes[0].set_ylabel("Portfolio Value ($)", fontsize=11)
                axes[0].legend(loc="upper left", framealpha=0.9)
                axes[0].grid(True, linestyle=":", alpha=0.6)

                # Underwater Drawdown
                axes[1].fill_between(x_axis, dd * 100, 0, color="#E53935", alpha=0.35)
                axes[1].plot(x_axis, dd * 100, color="#C62828", lw=1.2)
                axes[1].axhline(y=m["max_drawdown_pct"], color="#B71C1C", linestyle="--", lw=1.0, label=f"Max Drawdown ({m['max_drawdown_pct']:.2f}%)")
                axes[1].set_title("Underwater Portfolio Drawdown (%)", fontsize=11, fontweight="bold")
                axes[1].set_ylabel("Drawdown (%)", fontsize=10)
                axes[1].legend(loc="lower left", framealpha=0.9)
                axes[1].grid(True, linestyle=":", alpha=0.6)

                # PnL Distribution
                active_pnls = [t["pnl"] for t in calc.trades if t["signal"] != 0]
                if active_pnls:
                    bins = np.linspace(np.percentile(active_pnls, 1), np.percentile(active_pnls, 99), 35)
                    pos = [p for p in active_pnls if p >= 0]
                    neg = [p for p in active_pnls if p < 0]
                    axes[2].hist(pos, bins=bins, color="#43A047", alpha=0.75, label=f"Wins ({m['win_count']} | Avg: +${m['avg_win']:,.1f})", edgecolor="white")
                    axes[2].hist(neg, bins=bins, color="#E53935", alpha=0.75, label=f"Losses ({m['loss_count']} | Avg: -${m['avg_loss']:,.1f})", edgecolor="white")
                    axes[2].axvline(0, color="black", lw=1.0)
                    axes[2].axvline(m["expectancy"], color="#FB8C00", linestyle="--", lw=1.5, label=f"Expectancy: +${m['expectancy']:,.1f}/trade")
                    axes[2].set_title(f"Trade PnL Distribution (Win Rate: {m['win_rate']:.1f}% | RRR: {m['realized_rrr']:.2f})", fontsize=11, fontweight="bold")
                    axes[2].set_xlabel("Dollar PnL ($)", fontsize=10)
                    axes[2].legend(loc="upper right", framealpha=0.9)
                    axes[2].grid(True, linestyle=":", alpha=0.6)

                fig.tight_layout()
                filename = f"5_trading_dashboard_{split_label}.png" if "val" in split_label else f"6_trading_dashboard_{split_label}.png"
                fig.savefig(self.plots_folder / filename, dpi=200)
                plt.close(fig)
            except Exception as e:
                self._log(f"Warning: Failed to render trading plot {split_label}: {e}")

    def _save_metrics_and_summary(self):
        """Compile and save metrics.json and summary_report.txt."""
        # Clean serializable dictionary
        clean_metrics = {
            "target_column": self.target_column,
            "experiment_run": self.run_name,
            "created_at": datetime.now().isoformat(),
            "forecasting_metrics": self.final_metrics.get("forecasting", {}),
            "trading_strategies": {},
        }

        if self.ga_results:
            clean_metrics["ga_optimal_parameters"] = self.ga_results["best_params"]
            clean_metrics["ga_best_fitness"] = float(self.ga_results["best_fitness"])

        for strat_name, calc in self.trading_results.items():
            m = calc.metrics
            clean_metrics["trading_strategies"][strat_name] = {
                "initial_capital": m["initial_capital"],
                "final_equity": m["final_equity"],
                "strategy_return_pct": m["strategy_return_pct"],
                "benchmark_return_pct": m["benchmark_return_pct"],
                "alpha_pct": m["alpha_pct"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sortino_ratio": m["sortino_ratio"],
                "profit_factor": m["profit_factor"],
                "win_rate_pct": m["win_rate"],
                "total_trades": m["total_trades"],
                "win_count": m["win_count"],
                "loss_count": m["loss_count"],
                "avg_win": m["avg_win"],
                "avg_loss": m["avg_loss"],
                "realized_rrr": m["realized_rrr"],
                "expectancy_per_trade": m["expectancy"],
                "tp_hits": m["tp_hits"],
                "sl_hits": m["sl_hits"],
                "close_hits": m["close_hits"],
            }

        # Write metrics.json
        with open(self.run_folder / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2)

        # Generate Human-Readable Text Summary
        test_strat = clean_metrics["trading_strategies"].get("test_ga") or clean_metrics["trading_strategies"].get("test_baseline", {})
        fc_test = clean_metrics["forecasting_metrics"].get("test", {})

        report_lines = [
            "=" * 80,
            f"          QUANTITATIVE EXPERIMENT REPORT: {self.target_column.upper()}",
            "=" * 80,
            f"Run Directory    : {self.run_folder.resolve()}",
            f"Target Column    : {self.target_column}",
            f"Total Features   : {len(self.full_feature_cols)} features",
            f"LSTM Model       : {self.num_layers} layer(s), {self.hidden_size} hidden units, Lookback={self.window_size}",
            "-" * 80,
            "1. FORECASTING PERFORMANCE (TEST SPLIT):",
            f"  Directional Accuracy : {fc_test.get('directional_accuracy_pct', 0.0):.2f}%",
            f"  Return MAE           : {fc_test.get('return_mae', 0.0):.5f}",
            f"  Return RMSE          : {fc_test.get('return_rmse', 0.0):.5f}",
            f"  Price MAE            : {fc_test.get('price_mae', 0.0):.4f}",
            "-" * 80,
            "2. OUT-OF-SAMPLE TRADING STRATEGY (TEST SPLIT):",
            f"  Initial Capital      : ${test_strat.get('initial_capital', 10000):,.2f}",
            f"  Final Equity         : ${test_strat.get('final_equity', 0):,.2f}",
            f"  Strategy Return      : {test_strat.get('strategy_return_pct', 0):+.2f}%",
            f"  Buy & Hold Return    : {test_strat.get('benchmark_return_pct', 0):+.2f}%",
            f"  Alpha Generated      : {test_strat.get('alpha_pct', 0):+.2f}%",
            f"  Max Drawdown         : {test_strat.get('max_drawdown_pct', 0):.2f}%",
            f"  Sharpe Ratio         : {test_strat.get('sharpe_ratio', 0):.2f}",
            f"  Sortino Ratio        : {test_strat.get('sortino_ratio', 0):.2f}",
            f"  Profit Factor        : {test_strat.get('profit_factor', 0):.2f}",
            f"  Win Rate             : {test_strat.get('win_rate_pct', 0):.2f}% ({test_strat.get('win_count', 0)} wins / {test_strat.get('loss_count', 0)} losses)",
            f"  Total Trades         : {test_strat.get('total_trades', 0)} trades",
            f"  Trade Expectancy     : ${test_strat.get('expectancy_per_trade', 0):.2f} per trade",
            "-" * 80,
        ]

        if self.ga_results:
            report_lines.extend([
                "3. GENETIC ALGORITHM OPTIMAL PARAMETERS:",
                f"  Objective Metric   : {self.ga_fitness_metric.upper()} (Fitness: {clean_metrics['ga_best_fitness']:.4f})",
            ])
            for k, v in clean_metrics["ga_optimal_parameters"].items():
                if "ratio" in k or "leverage" in k:
                    report_lines.append(f"  {k:<18s} : {v:.2f}")
                else:
                    report_lines.append(f"  {k:<18s} : {v:.5f} ({v*100:.3f}%)")

        report_lines.extend([
            "-" * 80,
            "4. SAVED ARTIFACTS:",
            "  - model_best.pt                    (PyTorch trained model checkpoint)",
            "  - metrics.json                     (Complete machine-readable stats)",
            "  - predictions_{train,val,test}.csv (Predictions & price reconstructions)",
            "  - trading_trades_test.csv          (Trade-by-trade log for test set)",
            "  - plots/1_training_loss.png        (Loss curves over epochs)",
            "  - plots/2_test_returns_predictions.png",
            "  - plots/3_test_price_forecast.png",
            "  - plots/4_ga_convergence.png",
            "  - plots/5_trading_dashboard_val.png",
            "  - plots/6_trading_dashboard_test.png",
            "=" * 80,
        ])

        summary_text = "\n".join(report_lines)
        with open(self.run_folder / "summary_report.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)

        if self.verbose:
            print("\n" + summary_text)

    # ────────────────────────────────────────────────────────────────────────
    # 7. End-to-End Orchestrator
    # ────────────────────────────────────────────────────────────────────────
    def train(self):
        """Execute complete pipeline: Load -> Train -> Evaluate -> GA Optimize -> Save."""
        start_time = datetime.now()
        self._log(f"Starting GeneralTrainer pipeline for [{self.target_column}] at {start_time}")

        # Step 1: Data
        self.load_and_prepare_data()

        # Step 2: Dataloaders
        self.create_dataloaders()

        # Step 3: Model
        self.train_model()

        # Step 4: Evaluate
        self.evaluate()

        # Step 5: Trading & GA
        self.run_trading_and_ga()

        # Step 6: Save Artifacts
        self.save_results()

        elapsed = (datetime.now() - start_time).total_seconds()
        self._log(f"\nPipeline finished in {elapsed:.1f} seconds. Output: {self.run_folder.resolve()}")

        return {
            "run_folder": self.run_folder,
            "metrics": self.final_metrics,
            "evaluation": self.evaluation_results,
            "trading": self.trading_results,
            "ga_results": self.ga_results,
            "model": self.model,
        }


# ────────────────────────────────────────────────────────────────────────────
# One-Line Convenience Function
# ────────────────────────────────────────────────────────────────────────────
def train_timeseries_pipeline(
    data,
    target_column="Close",
    output_dir="runs",
    **kwargs,
):
    """
    Convenience function to train a directional forecasting model, run GA backtesting,
    and save all results to disk.

    Example
    -------
    >>> from general_trainer import train_timeseries_pipeline
    >>> results = train_timeseries_pipeline("FRED/sap500.csv", target_column="Close", output_dir="runs/SP500")
    """
    trainer = GeneralTrainer(
        data=data,
        target_column=target_column,
        output_dir=output_dir,
        **kwargs,
    )
    return trainer.train()


# ────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="General Time-Series Trainer & Trading Strategy Optimizer"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file or 'FRED' directory")
    parser.add_argument("--target", type=str, default="Close", help="Target price column name (e.g. Close, EUR, INR)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory to store results and plots")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs (default: 80)")
    parser.add_argument("--window_size", type=int, default=10, help="Lookback window sequence length (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--hidden_size", type=int, default=32, help="LSTM hidden units (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--no_ga", action="store_true", help="Disable Genetic Algorithm optimization")
    parser.add_argument("--ga_fitness", type=str, default="composite", choices=["composite", "return", "sharpe", "sortino", "calmar", "profit_factor"], help="GA optimization objective")
    parser.add_argument("--ga_pop", type=int, default=40, help="GA population size (default: 40)")
    parser.add_argument("--ga_gen", type=int, default=25, help="GA generations (default: 25)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial trading capital (default: 10000)")
    parser.add_argument("--train_on_val", action="store_true", help="Fine-tune model on validation data before testing")
    parser.add_argument("--val_epochs", type=int, default=15, help="Epochs to train on validation data (default: 15)")
    parser.add_argument("--val_lr", type=float, default=0.0001, help="Learning rate for validation training (default: 0.0001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    trainer = GeneralTrainer(
        data=args.data,
        target_column=args.target,
        output_dir=args.output_dir,
        epochs=args.epochs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        ga_optimize=not args.no_ga,
        ga_fitness_metric=args.ga_fitness,
        ga_population_size=args.ga_pop,
        ga_generations=args.ga_gen,
        initial_capital=args.capital,
        train_on_val=args.train_on_val,
        val_epochs=args.val_epochs,
        val_lr=args.val_lr,
        random_state=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    main()

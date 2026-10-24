#!/usr/bin/env python3
"""
================================================================================
REAL-WORLD INTRADAY TRADING SIMULATION WITH ROLLING MODEL RETRAINING (2026)
================================================================================
This script executes an authentic, leak-free live trading simulation for dates
following August 26, 2026 (the training boundary of the EUR/USD forecaster):

DATA ARCHITECTURE (FRED Spot Data vs Real-Time Market Data):
1. Training Baseline Data (FRED H.10 Federal Reserve Data):
   - Target / Asset: EUR (DEXUSEU) - U.S. Dollars to Euro Spot Exchange Rate
   - Foreign Currencies: INR (DEXINUS), JPY (DEXJPUS)
   - Macroeconomic Drivers: FFER (DFF - Federal Funds Effective Rate),
     US10YY (DGS10 - 10-Year Treasury Constant Maturity Yield)
   - Plus 18 Technical Indicators (MAs, EMAs, RSI, Bollinger Bands, Volatility, ATR proxies)
2. Live Market Ingestion (Yahoo Finance Real-Time Tickers):
   - EUR/USD Forex Spot: EURUSD=X (Global interbank spot matching FRED DEXUSEU)
   - USD/INR: INR=X | USD/JPY: JPY=X | US 10-Year Yield: ^TNX
   - Policy Rate: FFER forward-filled from last official Federal Reserve FOMC release
   - Technical Indicators: Identical 18 indicators calculated continuously.
   - Holiday Alignment: US bond holidays (e.g. Labor Day) are forward-filled so international
     forex sessions maintain valid continuous features without NaNs.
3. Intraday Execution Stream:
   - High-resolution 1-minute OHLC tick bars of EURUSD=X from Yahoo Finance.

ONLINE ROLLING WALK-FORWARD RETRAINING:
- Every N days (default: 10 trading days, bi-weekly interval):
  The model is fine-tuned on the accumulated newly ingested market data using
  BigMoveAwareLoss and a conservative learning rate (5e-5).
- This allows the model to continuously adapt its parameters and internal states
  to changing market regimes (e.g. shifts from consolidation to strong trends).
- Genetic Algorithm parameters (confidence deadzone threshold, TP/SL barriers)
  are dynamically re-calibrated on the updated model predictions.

EXECUTION FLOW:
1. Load Exported Model Checkpoint (`models/eur_lstm_final.pt`)
2. Continuous Daily Macro Integration (FRED + Yahoo Finance)
3. Initial Genetic Algorithm Calibration
4. Multi-Request Minute Intraday Cache (1h -> 5m -> 1m tiered chunking)
5. Day-by-Day Live Simulation with Online Rolling Retraining every N days
6. Institutional Performance Report, CSV Trade Log, JSON Metrics, and 3-Panel Plot
================================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution / scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils
from utils import (
    AdvancedProfitCalculator,
    BigMoveAwareLoss,
    GeneticStrategyOptimizer,
    calculate_technical_indicators,
    load_fred_data,
    load_model,
    optimize_strategy_ga,
    prepare_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate real-world trading using EUR LSTM model, rolling retraining, and yfinance minute streaming."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/eur_lstm_final.pt",
        help="Path to exported PyTorch model checkpoint (.pt).",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2026-08-27",
        help="Simulation start date (YYYY-MM-DD), default is day after training ends.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Simulation end date (YYYY-MM-DD). If None, runs through latest available market date.",
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=10000.0,
        help="Starting portfolio cash balance ($).",
    )
    parser.add_argument(
        "--fee_per_trade",
        type=float,
        default=0.0001,
        help="Transaction friction per trade in decimal (0.0001 = 1 pip spread + commission).",
    )
    parser.add_argument(
        "--retrain_interval",
        type=int,
        default=10,
        help="Interval in trading days to retrain/fine-tune the model on new ingested data (e.g. 5 or 10 days). Set 0 to disable.",
    )
    parser.add_argument(
        "--retrain_epochs",
        type=int,
        default=8,
        help="Number of epochs per rolling fine-tuning event.",
    )
    parser.add_argument(
        "--retrain_lr",
        type=float,
        default=5e-5,
        help="Fine-tuning learning rate (low rate avoids catastrophic forgetting).",
    )
    parser.add_argument(
        "--retrain_window",
        type=int,
        default=120,
        help="Lookback window in days of recent market history to train on during retraining.",
    )
    parser.add_argument(
        "--recalibrate_ga",
        action="store_true",
        default=True,
        help="Re-calibrate GA parameters on updated predictions after each model retraining event.",
    )
    parser.add_argument(
        "--use_saved_ga",
        action="store_true",
        default=False,
        help="Force using static GA parameters saved in checkpoint metadata instead of calibrating.",
    )
    parser.add_argument(
        "--ga_generations",
        type=int,
        default=25,
        help="Number of generations for GA calibration.",
    )
    parser.add_argument(
        "--ga_pop_size",
        type=int,
        default=40,
        help="Population size for GA calibration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/real_trading_2026",
        help="Directory where trade logs, metrics, and plots will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run PyTorch model ('cuda' or 'cpu'). Auto-detected if None.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print trade-by-trade live execution updates.",
    )
    return parser.parse_args()


# ==============================================================================
# 1. DATA INGESTION & FEATURE MATRIX INTEGRATION
# ==============================================================================
def build_continuous_feature_matrix(data_dir=None, start_date="2026-08-27"):
    """
    Build a continuous, updated macroeconomic and technical indicator matrix
    combining historical FRED series with recent daily market data from yfinance.

    Data Alignment:
    - FRED 'EUR' (DEXUSEU)   <---> yfinance 'EURUSD=X' (Forex spot rate)
    - FRED 'INR' (DEXINUS)   <---> yfinance 'INR=X' (USD/INR spot rate)
    - FRED 'JPY' (DEXJPUS)   <---> yfinance 'JPY=X' (USD/JPY spot rate)
    - FRED 'US10YY' (DGS10)  <---> yfinance '^TNX' (10-Year Treasury Yield)
    - FRED 'FFER' (DFF)      <---> Last official Fed funds rate (forward-filled)
    """
    data_dir = data_dir or (REPO_ROOT / "FRED")
    print(f"\n[1/5] Ingesting Historical Macro Data from: {data_dir.resolve()} ...")
    fred_raw = load_fred_data(data_dir)
    fred_df = prepare_data(fred_raw)
    print(f"  FRED Base Dataset (Federal Reserve H.10 Spot Rates):")
    print(f"    - Series: EUR (DEXUSEU), INR (DEXINUS), JPY (DEXJPUS), FFER (DFF), US10YY (DGS10)")
    print(f"    - Historical Records: {len(fred_df)} days (ending {fred_df.index[-1].strftime('%Y-%m-%d')})")

    # Dynamically adjust yf_start to precede start_date
    start_dt = pd.to_datetime(start_date)
    earliest_needed = (start_dt - timedelta(days=40)).strftime("%Y-%m-%d")
    yf_start = min("2026-08-01", earliest_needed)

    print(f"  Ingesting Real-Time Daily Market Quotes via yfinance from {yf_start} ...")
    print(f"    - Tickers: EURUSD=X (Spot), INR=X, JPY=X, ^TNX (10Y Yield)")
    yf_tickers = {
        "EUR": "EURUSD=X",
        "INR": "INR=X",
        "JPY": "JPY=X",
        "US10YY": "^TNX",
    }
    yf_raw = yf.download(
        list(yf_tickers.values()),
        start=yf_start,
        interval="1d",
        progress=False,
    )

    if isinstance(yf_raw.columns, pd.MultiIndex):
        yf_close = yf_raw["Close"]
    else:
        yf_close = yf_raw

    daily_updates = pd.DataFrame(index=pd.to_datetime(yf_close.index).tz_localize(None))
    for col, ticker in yf_tickers.items():
        if ticker in yf_close.columns:
            daily_updates[col] = yf_close[ticker].values

    # Clean missing weekend/holiday rows
    daily_updates = daily_updates.dropna(subset=["EUR"])

    # Forward-fill FFER from the last official FRED release
    daily_updates["FFER"] = fred_df["FFER"].iloc[-1]

    # Combine FRED base data with newer daily updates
    new_dates_mask = ~daily_updates.index.isin(fred_df.index)
    newer_rows = daily_updates[new_dates_mask]
    combined_macro = pd.concat([fred_df, newer_rows]).sort_index()
    combined_macro = combined_macro[~combined_macro.index.duplicated(keep="last")]

    # Forward-fill holiday gaps (e.g. US bond market closed on Labor Day while European forex trades)
    combined_macro = combined_macro.ffill().bfill()

    print(f"  Combined Continuous Macro Dataset: {len(combined_macro)} days ({combined_macro.index[0].strftime('%Y-%m-%d')} to {combined_macro.index[-1].strftime('%Y-%m-%d')})")

    # Compute full set of 23 technical indicators for EUR
    eur_feature_df = calculate_technical_indicators(combined_macro, target_column="EUR")
    eur_feature_df["TARGET_LOG_RETURN_NEXT"] = eur_feature_df["Log_Returns"].shift(-1)
    eur_feature_df = eur_feature_df.dropna(subset=["Log_Returns", "MA_100", "RSI"])
    print(f"  Constructed Technical Indicators: {eur_feature_df.shape[1]} features, {len(eur_feature_df)} total rows.")
    return eur_feature_df


# ==============================================================================
# 2. INTRADAY MINUTE DATA STREAMING PROVIDER (MULTI-REQUEST CHUNKING)
# ==============================================================================
class MinuteDataStreamer:
    """
    Provides minute-level OHLC price bars for EUR/USD simulated streaming.
    Overcomes Yahoo Finance API limits by making multiple chunked requests across
    adaptive tiers based on date recency:
    - Tier 1 (>60 days ago): 1-hour data in 15-day chunks resampled to 1-minute steps
    - Tier 2 (30-60 days ago): 5-minute data in 7-day chunks resampled to 1-minute steps
    - Tier 3 (<30 days ago): Native 1-minute data in 5-day chunks
    All bars are unified into a contiguous, high-resolution minute database.
    """
    def __init__(self, start_date="2026-08-27", ticker="EURUSD=X"):
        self.ticker = ticker
        self.start_date = start_date
        print(f"\n[2/5] Initializing Multi-Request Minute Data Feed for {self.ticker} from {start_date} ...")
        self.bars = self._fetch_all_intraday_data(start_date)

    def _fetch_all_intraday_data(self, start_date):
        now = pd.Timestamp.now()
        start_dt = pd.to_datetime(start_date)
        limit_60d = (now - timedelta(days=58)).normalize()
        limit_30d = (now - timedelta(days=28)).normalize()

        all_bars = []
        t0 = datetime.now()

        # Tier 1: Older than 60 days (1h data in 15-day chunks)
        if start_dt < limit_60d:
            cur_dt = start_dt
            end_t1 = limit_60d
            print(f"  [Tier 1] Fetching 1h data in 15-day chunks ({cur_dt.strftime('%Y-%m-%d')} to {end_t1.strftime('%Y-%m-%d')})...")
            while cur_dt < end_t1:
                next_dt = min(cur_dt + timedelta(days=15), end_t1)
                s_str = cur_dt.strftime("%Y-%m-%d")
                e_str = next_dt.strftime("%Y-%m-%d")
                if s_str == e_str:
                    break
                try:
                    df = yf.download(self.ticker, start=s_str, end=e_str, interval="1h", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if not df.empty:
                        df_1m = df.resample("1min").ffill()
                        all_bars.append(df_1m)
                except Exception as e:
                    print(f"    Warning: 1h chunk {s_str}->{e_str} failed: {e}")
                cur_dt = next_dt

        # Tier 2: Between 60 and 30 days ago (5m data in 7-day chunks)
        cur_dt = max(start_dt, limit_60d)
        end_t2 = min(now, limit_30d)
        if cur_dt < end_t2:
            print(f"  [Tier 2] Fetching 5m data in 7-day chunks ({cur_dt.strftime('%Y-%m-%d')} to {end_t2.strftime('%Y-%m-%d')})...")
            while cur_dt < end_t2:
                next_dt = min(cur_dt + timedelta(days=7), end_t2)
                s_str = cur_dt.strftime("%Y-%m-%d")
                e_str = next_dt.strftime("%Y-%m-%d")
                if s_str == e_str:
                    break
                try:
                    df = yf.download(self.ticker, start=s_str, end=e_str, interval="5m", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if not df.empty:
                        df_1m = df.resample("1min").ffill()
                        all_bars.append(df_1m)
                except Exception as e:
                    print(f"    Warning: 5m chunk {s_str}->{e_str} failed: {e}")
                cur_dt = next_dt

        # Tier 3: Last 30 days (Native 1m data in 5-day chunks)
        cur_dt = max(start_dt, limit_30d)
        if cur_dt < now:
            print(f"  [Tier 3] Fetching native 1m data in 5-day chunks ({cur_dt.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')})...")
            while cur_dt < now:
                next_dt = min(cur_dt + timedelta(days=5), now)
                s_str = cur_dt.strftime("%Y-%m-%d")
                e_str = next_dt.strftime("%Y-%m-%d")
                if s_str == e_str:
                    break
                try:
                    df = yf.download(self.ticker, start=s_str, end=e_str, interval="1m", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if not df.empty:
                        all_bars.append(df)
                    else:
                        df_5 = yf.download(self.ticker, start=s_str, end=e_str, interval="5m", progress=False)
                        if isinstance(df_5.columns, pd.MultiIndex):
                            df_5.columns = df_5.columns.get_level_values(0)
                        if not df_5.empty:
                            all_bars.append(df_5.resample("1min").ffill())
                except Exception as e:
                    print(f"    Warning: 1m chunk {s_str}->{e_str} failed: {e}")
                cur_dt = next_dt

        if all_bars:
            combined = pd.concat(all_bars).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  Multi-Request Complete: {len(combined):,} continuous 1-minute bars cached in {elapsed:.2f}s.")
            print(f"  Intraday Range Cached : {combined.index[0]} to {combined.index[-1]}")
            return combined
        else:
            print("  Warning: No intraday bars could be fetched.")
            return pd.DataFrame()

    def get_day_stream(self, date_str):
        """
        Return a generator yielding minute bars (as dicts) for the specified date.
        """
        if self.bars.empty:
            return

        sub = self.bars[self.bars.index.strftime("%Y-%m-%d") == date_str]
        if sub.empty:
            return

        for idx, row in sub.iterrows():
            yield {
                "timestamp": idx,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }


# ==============================================================================
# 3. ONLINE ROLLING MODEL RETRAINING ENGINE
# ==============================================================================
def fine_tune_model_on_recent_data(
    model,
    data_slice,
    feature_cols,
    window_size,
    feature_scaler,
    target_scaler,
    device,
    epochs=8,
    lr=5e-5,
):
    """
    Fine-tunes the PyTorch model on recently elapsed market days.
    Input data slice contains completed sessions up to day T-1 with realized targets.
    """
    valid_data = data_slice.dropna(subset=["TARGET_LOG_RETURN_NEXT"])
    if len(valid_data) <= window_size + 5:
        return 0.0

    feats = valid_data[feature_cols].values
    scaled_feats = feature_scaler.transform(feats)
    targets = valid_data["TARGET_LOG_RETURN_NEXT"].values.reshape(-1, 1)
    scaled_targets = target_scaler.transform(targets).flatten()

    X_list, y_list = [], []
    for i in range(window_size, len(valid_data)):
        X_list.append(scaled_feats[i - window_size : i])
        y_list.append(scaled_targets[i - 1])

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = BigMoveAwareLoss(dir_penalty=0.5, mag_penalty=2.0, sigma_threshold=1.5)

    model.train()
    total_loss = 0.0
    for ep in range(epochs):
        ep_loss = 0.0
        n_b = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx).view(-1)
            loss = loss_fn(out, by.view(-1))
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_b += 1
        total_loss = ep_loss / max(n_b, 1)

    model.eval()
    return total_loss


# ==============================================================================
# 4. GENETIC ALGORITHM STRATEGY OPTIMIZATION
# ==============================================================================
def calibrate_strategy_parameters(
    model_bundle,
    eur_df,
    calibration_end_date="2026-08-26",
    lookback_samples=500,
    generations=25,
    population_size=40,
    use_saved=False,
    verbose=True,
):
    """
    Run Genetic Algorithm optimization on historical model predictions to discover
    optimal confidence deadzone, stop-loss, take-profit, and risk parameters.
    """
    saved_ga = model_bundle.get("metadata", {}).get("ga_optimal_params", {})
    if use_saved and saved_ga:
        if verbose:
            print("  Using Pre-Calibrated Optimal GA Parameters from Checkpoint Metadata:")
            for k, v in saved_ga.items():
                print(f"    - {k:<18}: {v}")
        return saved_ga

    feature_cols = model_bundle["feature_columns"]
    window_size = model_bundle["window_size"]
    predict_fn = model_bundle["predict"]

    pre_split_dates = [d for d in eur_df.index if d.strftime("%Y-%m-%d") <= calibration_end_date]
    if len(pre_split_dates) < window_size + 20:
        pre_split_dates = list(eur_df.index[:lookback_samples])

    calib_dates = pre_split_dates[-lookback_samples:]
    calib_preds = []
    calib_actuals = []

    for d in calib_dates:
        loc = eur_df.index.get_loc(d)
        if loc >= window_size:
            window = eur_df.iloc[loc - window_size : loc][feature_cols]
            pred = predict_fn(window)
            actual = eur_df.loc[d, "Log_Returns"]
            calib_preds.append(pred)
            calib_actuals.append(actual)

    calib_preds = np.asarray(calib_preds, dtype=np.float64)
    calib_actuals = np.asarray(calib_actuals, dtype=np.float64)

    if verbose:
        print(f"  Calibration Sample Window : {len(calib_preds)} days (ending {calibration_end_date})")
        print(f"  Running GA Optimization ({generations} Generations, Pop Size {population_size}, Objective: Calmar) ...")

    ga_result = optimize_strategy_ga(
        predictions=calib_preds,
        actual_returns=calib_actuals,
        fitness_metric="calmar",
        generations=generations,
        population_size=population_size,
        crossover_prob=0.8,
        mutation_prob=0.2,
        initial_capital=10000.0,
        fee_per_trade=0.0001,
        verbose=False,
    )

    best_params = {k: float(v) for k, v in ga_result["best_params"].items()}
    if verbose:
        print(f"  GA Calibration Completed! Best Fitness Achieved: {ga_result['best_fitness']:.4f}")
        for k, v in best_params.items():
            if "ratio" in k or "leverage" in k:
                print(f"    - {k:<18}: {v:.2f}")
            elif "risk_per_trade" in k:
                print(f"    - {k:<18}: {v * 100:.3f}% of capital")
            else:
                print(f"    - {k:<18}: {v:.5f} ({v * 100:.3f}%)")

    return best_params


# ==============================================================================
# 5. REAL-TIME MINUTE SIMULATION WITH ROLLING RETRAINING
# ==============================================================================
def run_simulation(
    model_bundle,
    eur_df,
    streamer,
    strategy_params,
    start_date="2026-08-27",
    end_date=None,
    initial_capital=10000.0,
    fee_per_trade=0.0001,
    retrain_interval=10,
    retrain_epochs=8,
    retrain_lr=5e-5,
    retrain_window=120,
    recalibrate_ga_on_retrain=True,
    verbose=True,
):
    """
    Simulates real-world trade execution on minute-level data day-by-day,
    with scheduled rolling walk-forward retraining on newly ingested data.
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_columns"]
    window_size = model_bundle["window_size"]
    feature_scaler = model_bundle["feature_scaler"]
    target_scaler = model_bundle["target_scaler"]
    device = next(model.parameters()).device
    predict_fn = model_bundle["predict"]

    all_dates = [d for d in eur_df.index if d.strftime("%Y-%m-%d") >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d.strftime("%Y-%m-%d") <= end_date]

    print(f"\n[4/5] Executing Real-World Minute Trading Simulation ({len(all_dates)} Trading Days) ...")
    print("=" * 86)
    print(f"  Starting Capital   : ${initial_capital:,.2f}")
    if retrain_interval:
        print(f"  Retraining Interval: Every {retrain_interval} Days (Epochs: {retrain_epochs}, LR: {retrain_lr})")
    else:
        print(f"  Retraining Interval: Disabled (Static Inference)")
    print(f"  Take-Profit (TP)   : {strategy_params['take_profit'] * 100:+.3f}% | Stop-Loss (SL): {strategy_params['stop_loss'] * 100:-.3f}%")
    print(f"  Confidence Thresh  : {strategy_params['threshold']:.5f} | Risk/Trade: {strategy_params['risk_per_trade'] * 100:.2f}% | Max Leverage: {strategy_params['max_leverage']:.1f}x")
    print("=" * 86)

    capital = initial_capital
    trades = []
    benchmark_equity = [initial_capital]
    initial_eur_price = None
    retraining_events = []

    for step_idx, current_date in enumerate(all_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        loc = eur_df.index.get_loc(current_date)

        # ----------------------------------------------------------------------
        # Rolling Walk-Forward Retraining on Newly Ingested Market Data
        # ----------------------------------------------------------------------
        if retrain_interval and step_idx > 0 and step_idx % retrain_interval == 0:
            print("\n" + "-" * 75)
            print(f"  [ONLINE RETRAINING] Step {step_idx} | Date: {date_str}")
            print(f"  Fine-tuning LSTM on past {retrain_window} market days (strictly preceding {date_str}) ...")

            train_slice = eur_df.iloc[max(0, loc - retrain_window) : loc]
            loss_val = fine_tune_model_on_recent_data(
                model=model,
                data_slice=train_slice,
                feature_cols=feature_cols,
                window_size=window_size,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                device=device,
                epochs=retrain_epochs,
                lr=retrain_lr,
            )
            print(f"  Fine-Tuning Loss: {loss_val:.5f} across {retrain_epochs} epochs.")

            if recalibrate_ga_on_retrain:
                calib_end_yesterday = eur_df.index[loc - 1].strftime("%Y-%m-%d")
                print(f"  Re-calibrating GA Strategy Parameters on updated model representations...")
                new_params = calibrate_strategy_parameters(
                    model_bundle=model_bundle,
                    eur_df=eur_df,
                    calibration_end_date=calib_end_yesterday,
                    lookback_samples=250,
                    generations=15,
                    population_size=30,
                    verbose=False,
                )
                strategy_params = new_params
                print(f"  Updated Parameters: Thresh={strategy_params['threshold']:.5f}, SL={strategy_params['stop_loss']*100:.2f}%, TP={strategy_params['take_profit']*100:.2f}%, Risk={strategy_params['risk_per_trade']*100:.2f}%")

            retraining_events.append({
                "step": step_idx,
                "date": date_str,
                "loss": loss_val,
                "updated_params": dict(strategy_params),
            })
            print("-" * 75 + "\n")

        # Current Strategy Parameters
        tp_pct = strategy_params["take_profit"]
        sl_pct = strategy_params["stop_loss"]
        threshold = strategy_params["threshold"]
        risk_per_trade = strategy_params["risk_per_trade"]
        max_leverage = strategy_params["max_leverage"]

        # Strict no-leakage: input window ends at loc (exclusive of current day)
        window = eur_df.iloc[loc - window_size : loc][feature_cols]
        predicted_return = float(predict_fn(window))

        # Track Benchmark Buy & Hold on EUR
        day_actual_return = float(eur_df.loc[current_date, "Log_Returns"])
        benchmark_capital = benchmark_equity[-1] * (1.0 + day_actual_return)
        benchmark_equity.append(benchmark_capital)

        # Pull day's minute stream generator
        minute_stream = list(streamer.get_day_stream(date_str))
        if not minute_stream:
            continue

        first_bar = minute_stream[0]
        entry_price = first_bar["open"]
        if initial_eur_price is None:
            initial_eur_price = entry_price

        # Signal Derivation
        if predicted_return > threshold:
            signal = 1
            direction = "LONG"
        elif predicted_return < -threshold:
            signal = -1
            direction = "SHORT"
        else:
            signal = 0
            direction = "NEUTRAL"

        # Neutral day - No trade executed
        if signal == 0:
            trades.append({
                "date": date_str,
                "signal": 0,
                "direction": "NEUTRAL",
                "predicted_return": predicted_return,
                "entry_time": first_bar["timestamp"],
                "entry_price": entry_price,
                "exit_time": minute_stream[-1]["timestamp"],
                "exit_price": minute_stream[-1]["close"],
                "exit_reason": "NO_TRADE",
                "duration_min": 0,
                "position_size": 0.0,
                "leverage": 0.0,
                "trade_return": 0.0,
                "pnl": 0.0,
                "capital": capital,
                "benchmark": benchmark_capital,
            })
            if verbose:
                print(f"[{date_str}] Signal: NEUTRAL  | Pred: {predicted_return:+.5f} (Below Thresh {threshold:.5f}) | No Trade")
            continue

        # Position Sizing: Fixed Fractional Risk
        pos_size = (capital * risk_per_trade) / max(sl_pct, 1e-5)
        pos_size = min(pos_size, capital * max_leverage)
        effective_leverage = pos_size / capital

        # Define Price Barriers
        if signal == 1:
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)
        else:
            tp_price = entry_price * (1.0 - tp_pct)
            sl_price = entry_price * (1.0 + sl_pct)

        exit_reason = "EOD_CLOSE"
        exit_price = minute_stream[-1]["close"]
        exit_time = minute_stream[-1]["timestamp"]
        duration_minutes = len(minute_stream)

        # Minute-by-Minute Streaming Simulation
        for bar_idx, bar in enumerate(minute_stream):
            b_high = bar["high"]
            b_low = bar["low"]
            b_time = bar["timestamp"]

            if signal == 1:  # LONG
                if b_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_time = b_time
                    duration_minutes = bar_idx + 1
                    break
                elif b_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_time = b_time
                    duration_minutes = bar_idx + 1
                    break
            else:  # SHORT
                if b_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_time = b_time
                    duration_minutes = bar_idx + 1
                    break
                elif b_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_time = b_time
                    duration_minutes = bar_idx + 1
                    break

        # Calculate Returns and PnL
        if signal == 1:
            raw_trade_return = (exit_price - entry_price) / entry_price
        else:
            raw_trade_return = (entry_price - exit_price) / entry_price

        net_trade_return = raw_trade_return - fee_per_trade
        dollar_pnl = pos_size * net_trade_return
        capital = max(capital + dollar_pnl, 0.0)

        trades.append({
            "date": date_str,
            "signal": signal,
            "direction": direction,
            "predicted_return": predicted_return,
            "entry_time": first_bar["timestamp"],
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "duration_min": duration_minutes,
            "position_size": pos_size,
            "leverage": effective_leverage,
            "trade_return": net_trade_return,
            "pnl": dollar_pnl,
            "capital": capital,
            "benchmark": benchmark_capital,
        })

        if verbose:
            color = "+" if dollar_pnl >= 0 else "-"
            print(
                f"[{date_str}] {direction:<5} | Entry: {entry_price:.5f} -> Exit: {exit_price:.5f} ({exit_reason:<9}) | "
                f"PnL: {color}${abs(dollar_pnl):>7.2f} ({net_trade_return * 100:>+6.2f}%) | "
                f"Equity: ${capital:>9.2f}"
            )

    trades_df = pd.DataFrame(trades)
    return trades_df, initial_capital, benchmark_equity, retraining_events


# ==============================================================================
# 6. INSTITUTIONAL METRICS CALCULATION & REPORTING
# ==============================================================================
def calculate_institutional_metrics(trades_df, initial_capital):
    """
    Compute comprehensive institutional quantitative trading metrics.
    """
    final_equity = trades_df["capital"].iloc[-1]
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    final_bench = trades_df["benchmark"].iloc[-1]
    bench_return_pct = ((final_bench - initial_capital) / initial_capital) * 100.0
    alpha_pct = net_return_pct - bench_return_pct

    num_days = len(trades_df)
    cagr_pct = (((final_equity / initial_capital) ** (252.0 / max(num_days, 1))) - 1.0) * 100.0

    equity_series = trades_df["capital"].values
    running_max = np.maximum.accumulate(equity_series)
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown_pct = float(np.min(drawdowns) * 100.0)

    active_trades = trades_df[trades_df["signal"] != 0]
    total_active = len(active_trades)
    longs = len(trades_df[trades_df["signal"] == 1])
    shorts = len(trades_df[trades_df["signal"] == -1])
    neutrals = len(trades_df[trades_df["signal"] == 0])

    if total_active > 0:
        winning_trades = active_trades[active_trades["pnl"] > 0]
        losing_trades = active_trades[active_trades["pnl"] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_active) * 100.0

        gross_profit = float(winning_trades["pnl"].sum())
        gross_loss = float(abs(losing_trades["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = float(winning_trades["pnl"].mean()) if win_count > 0 else 0.0
        avg_loss = float(abs(losing_trades["pnl"].mean())) if loss_count > 0 else 0.0
        realized_rrr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss)

        returns = active_trades["trade_return"].values
        ret_mean = np.mean(returns)
        ret_std = np.std(returns) if len(returns) > 1 else 1e-6
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-6

        sharpe_ratio = float((ret_mean / ret_std) * np.sqrt(252)) if ret_std > 0 else 0.0
        sortino_ratio = float((ret_mean / downside_std) * np.sqrt(252)) if downside_std > 0 else 0.0
        calmar_ratio = float(cagr_pct / abs(max_drawdown_pct)) if max_drawdown_pct < 0 else 0.0
    else:
        win_count, loss_count, win_rate = 0, 0, 0.0
        gross_profit, gross_loss, profit_factor = 0.0, 0.0, 0.0
        avg_win, avg_loss, realized_rrr, expectancy = 0.0, 0.0, 0.0, 0.0
        sharpe_ratio, sortino_ratio, calmar_ratio = 0.0, 0.0, 0.0

    exit_counts = trades_df["exit_reason"].value_counts().to_dict()

    metrics = {
        "initial_capital": initial_capital,
        "final_portfolio_value": final_equity,
        "strategy_net_return_pct": net_return_pct,
        "benchmark_return_pct": bench_return_pct,
        "strategy_alpha_pct": alpha_pct,
        "annualized_cagr_pct": cagr_pct,
        "total_days_evaluated": num_days,
        "total_trades_executed": total_active,
        "long_trades": longs,
        "short_trades": shorts,
        "neutral_days": neutrals,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate_pct": win_rate,
        "average_win_dollar": avg_win,
        "average_loss_dollar": avg_loss,
        "realized_risk_reward": realized_rrr,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "trade_expectancy": expectancy,
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_sharpe": sharpe_ratio,
        "annualized_sortino": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "exit_reasons": exit_counts,
        "drawdowns": drawdowns.tolist(),
    }
    return metrics


def print_institutional_report(m, retrain_interval):
    """
    Print the institutional report in clean ASCII format.
    """
    print("\n" + "=" * 80)
    print("           INSTITUTIONAL QUANT TRADING PERFORMANCE REPORT (2026)")
    print("=" * 80)
    print(f"CONFIGURATION: Online Rolling Retrain = {'Every ' + str(retrain_interval) + ' Days' if retrain_interval else 'Disabled'}")
    print("-" * 80)
    print("CAPITAL & RETURNS:")
    print(f"  Initial Capital        : ${m['initial_capital']:,.2f}")
    print(f"  Final Portfolio Value  : ${m['final_portfolio_value']:,.2f}")
    print(f"  Strategy Net Return    : {m['strategy_net_return_pct']:+.2f}%")
    print(f"  Benchmark (Buy & Hold) : {m['benchmark_return_pct']:+.2f}%")
    print(f"  Strategy Alpha         : {m['strategy_alpha_pct']:+.2f}%")
    print(f"  Annualized CAGR        : {m['annualized_cagr_pct']:+.2f}%")
    print("-" * 80)
    print("TRADE EXECUTION & WIN/LOSS METRICS:")
    print(f"  Total Trades Executed  : {m['total_trades_executed']} (Long: {m['long_trades']}, Short: {m['short_trades']}, Neutral: {m['neutral_days']})")
    print(f"  Winning Trades         : {m['winning_trades']} ({m['win_rate_pct']:.2f}%)")
    print(f"  Losing Trades          : {m['losing_trades']} ({100.0 - m['win_rate_pct']:.2f}%)")
    print(f"  Average Win ($)        : ${m['average_win_dollar']:,.2f}")
    print(f"  Average Loss ($)       : ${m['average_loss_dollar']:,.2f}")
    print(f"  Realized Risk/Reward   : {m['realized_risk_reward']:.2f}")
    print(f"  Gross Profit           : ${m['gross_profit']:,.2f}")
    print(f"  Gross Loss             : ${m['gross_loss']:,.2f}")
    print(f"  Profit Factor          : {m['profit_factor']:.2f}")
    print(f"  Trade Expectancy       : ${m['trade_expectancy']:,.2f} per trade")
    print("-" * 80)
    print("RISK-ADJUSTED METRICS:")
    print(f"  Max Drawdown           : {m['max_drawdown_pct']:.2f}%")
    print(f"  Annualized Sharpe      : {m['annualized_sharpe']:.2f}")
    print(f"  Annualized Sortino     : {m['annualized_sortino']:.2f}")
    print(f"  Calmar Ratio           : {m['calmar_ratio']:.2f}")
    print("-" * 80)
    print("EXIT REASONS BREAKDOWN:")
    for reason, count in m["exit_reasons"].items():
        print(f"  {reason:<22} : {count}")
    print("=" * 80 + "\n")


# ==============================================================================
# 7. VISUAL ANALYTICS DASHBOARD
# ==============================================================================
def plot_dashboard(trades_df, metrics, output_path, retrain_events=None):
    """
    Generate professional 3-panel quant trading performance dashboard:
    1) Equity Curve vs Buy & Hold Benchmark (with Retraining Markers)
    2) Underwater Portfolio Drawdown (%)
    3) Trade PnL Distribution & Win/Loss Counts
    """
    dates = pd.to_datetime(trades_df["date"])
    equity = trades_df["capital"].values
    benchmark = trades_df["benchmark"].values
    drawdowns = np.array(metrics["drawdowns"]) * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2.2, 1.2, 1.4]})

    # Panel 1: Cumulative Equity vs Benchmark
    axes[0].plot(
        dates,
        equity,
        label=f"Strategy Equity (${metrics['final_portfolio_value']:,.0f} / {metrics['strategy_net_return_pct']:+.1f}%)",
        color="#1E88E5",
        linewidth=2.4,
    )
    axes[0].plot(
        dates,
        benchmark,
        label=f"Buy & Hold EUR (${benchmark[-1]:,.0f} / {metrics['benchmark_return_pct']:+.1f}%)",
        color="#78909C",
        linestyle="--",
        linewidth=1.6,
    )
    axes[0].axhline(
        y=metrics["initial_capital"],
        color="#B0BEC5",
        linestyle=":",
        linewidth=1.2,
        label=f"Initial Capital (${metrics['initial_capital']:,.0f})",
    )

    # Highlight Retraining Milestones
    if retrain_events:
        for idx, ev in enumerate(retrain_events):
            ev_date = pd.to_datetime(ev["date"])
            lbl = "Model Retraining Event" if idx == 0 else None
            axes[0].axvline(x=ev_date, color="#8E24AA", linestyle="--", alpha=0.65, label=lbl)

    axes[0].set_title(
        f"Real-World Minute Simulation with Online Rolling Retraining\n"
        f"Alpha: {metrics['strategy_alpha_pct']:+.2f}% | Profit Factor: {metrics['profit_factor']:.2f} | Sharpe: {metrics['annualized_sharpe']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("Portfolio Value ($)", fontsize=11)
    axes[0].legend(loc="upper left", framealpha=0.9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Underwater Drawdown
    axes[1].fill_between(dates, drawdowns, 0, color="#E53935", alpha=0.35)
    axes[1].plot(dates, drawdowns, color="#C62828", linewidth=1.4)
    axes[1].axhline(
        y=metrics["max_drawdown_pct"],
        color="#B71C1C",
        linestyle="--",
        linewidth=1.1,
        label=f"Max Drawdown ({metrics['max_drawdown_pct']:.2f}%)",
    )
    axes[1].set_title("Underwater Portfolio Drawdown (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Drawdown (%)", fontsize=10)
    axes[1].legend(loc="lower left", framealpha=0.9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Trade PnL Distribution
    active_pnls = trades_df[trades_df["signal"] != 0]["pnl"].values
    if len(active_pnls) > 0:
        pos_pnls = [p for p in active_pnls if p >= 0]
        neg_pnls = [p for p in active_pnls if p < 0]
        min_p = min(active_pnls)
        max_p = max(active_pnls)
        bins = np.linspace(min_p * 1.1, max_p * 1.1, 25) if min_p != max_p else 10

        axes[2].hist(
            pos_pnls,
            bins=bins,
            color="#43A047",
            alpha=0.8,
            label=f"Wins ({metrics['winning_trades']} | Avg: +${metrics['average_win_dollar']:,.1f})",
            edgecolor="white",
        )
        axes[2].hist(
            neg_pnls,
            bins=bins,
            color="#E53935",
            alpha=0.8,
            label=f"Losses ({metrics['losing_trades']} | Avg: -${metrics['average_loss_dollar']:,.1f})",
            edgecolor="white",
        )
        axes[2].axvline(x=0, color="black", linewidth=1.0)
        axes[2].axvline(
            x=metrics["trade_expectancy"],
            color="#FB8C00",
            linestyle="--",
            linewidth=1.6,
            label=f"Expectancy: +${metrics['trade_expectancy']:,.1f}/trade",
        )
        axes[2].set_title(
            f"Trade PnL Distribution (Realized RRR: {metrics['realized_risk_reward']:.2f} | Win Rate: {metrics['win_rate_pct']:.1f}%)",
            fontsize=11,
            fontweight="bold",
        )
        axes[2].set_xlabel("Dollar PnL ($)", fontsize=10)
        axes[2].set_ylabel("Trade Count", fontsize=10)
        axes[2].legend(loc="upper right", framealpha=0.9)
        axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Visual Performance Dashboard saved to: {output_path.resolve()}")


# ==============================================================================
# MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("  EUROPEAN FOREX (EUR/USD) REAL-WORLD MINUTE TRADING SIMULATION")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 86)

    # Step 1: Load Model Checkpoint
    model_path = Path(args.model_path)
    print(f"\n[Step 1] Loading Trained Model Checkpoint: {model_path.resolve()} ...")
    model_bundle = load_model(model_path, device=args.device, verbose=True)

    # Step 2: Build Updated Macro & Technical Indicator Matrix
    eur_df = build_continuous_feature_matrix(start_date=args.start_date)

    # Determine calibration end date (day before simulation start)
    sim_start_dt = pd.to_datetime(args.start_date)
    calib_end_date = (sim_start_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # Step 3: Calibrate Quantitative Parameters via Genetic Algorithm
    strategy_params = calibrate_strategy_parameters(
        model_bundle=model_bundle,
        eur_df=eur_df,
        calibration_end_date=calib_end_date,
        lookback_samples=500,
        generations=args.ga_generations,
        population_size=args.ga_pop_size,
        use_saved=args.use_saved_ga,
    )

    # Step 4: Initialize Minute Level Streaming Provider (Multi-Request Chunking)
    streamer = MinuteDataStreamer(start_date=args.start_date, ticker="EURUSD=X")

    # Step 5: Execute Real-Time Minute Simulation with Rolling Retraining
    trades_df, init_cap, benchmark_equity, retrain_events = run_simulation(
        model_bundle=model_bundle,
        eur_df=eur_df,
        streamer=streamer,
        strategy_params=strategy_params,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        fee_per_trade=args.fee_per_trade,
        retrain_interval=args.retrain_interval,
        retrain_epochs=args.retrain_epochs,
        retrain_lr=args.retrain_lr,
        retrain_window=args.retrain_window,
        recalibrate_ga_on_retrain=args.recalibrate_ga,
        verbose=args.verbose,
    )

    # Step 6: Compute Institutional Performance Metrics & Print Report
    metrics = calculate_institutional_metrics(trades_df, init_cap)
    print_institutional_report(metrics, args.retrain_interval)

    # Step 7: Export Logs and Visualizations
    csv_path = output_dir / "simulation_2026_trades.csv"
    json_path = output_dir / "simulation_2026_metrics.json"
    plot_path = output_dir / "simulation_2026_performance.png"

    # Save Trade Log CSV
    trades_df.to_csv(csv_path, index=False)
    print(f"[Step 7] Artifacts Exported:")
    print(f"  Trade-by-Trade Execution Log : {csv_path.resolve()}")

    # Save Metrics JSON (strip numpy arrays)
    metrics_export = {k: v for k, v in metrics.items() if k != "drawdowns"}
    metrics_export["strategy_parameters"] = strategy_params
    metrics_export["retraining_events"] = retrain_events
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"  Performance Metrics JSON     : {json_path.resolve()}")

    # Generate and Save Performance Dashboard Plot
    plot_dashboard(trades_df, metrics, plot_path, retrain_events)
    print("\nSimulation successfully completed!")


if __name__ == "__main__":
    main()

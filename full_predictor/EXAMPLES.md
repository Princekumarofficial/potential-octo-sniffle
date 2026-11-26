# Examples

## Train and evaluate HYUNDAI

Run the example from inside `full_predictor` using a Python environment with NumPy, pandas, PyTorch, pyarrow, arch, matplotlib, and scikit-learn installed. The script reads `processed_datasets/NIFTY50/HYUNDAI.parquet`. If that file is missing, follow the [transformation guide](docs/DATA_TRANSFORMATION_FLOW.md) to generate it.

```powershell
python .\main.py
```

The settings are defined near the top of [main.py](main.py). The script uses CUDA when PyTorch detects it and otherwise runs on CPU.

1. **Load and split.** Load HYUNDAI and divide whole sessions chronologically into roughly 80% training, 10% validation, and 10% test.
2. **Build windows.** Recompute indicators from observed bars, then select 30 consecutive bars with finite input features and an observed next-minute target in the same segment. The target is the simple return from the last input close to the next close. Only training batches are shuffled.
3. **Train.** Fit `LSTMForecaster` with `BigMoveAwareLoss` and Adam for up to 20 epochs. Early stopping uses validation loss and restores the best validation weights.
4. **Evaluate.** Print test loss, MAE, MSE, RMSE, directional accuracy, and up/down recall. Compare directional accuracy with the always-up baseline. The direction metric counts zero returns as up.
5. **Train the trading policy.** Three expanding LSTM folds predict later blocks within the training split. Those forecasts train a tabular Q-learning agent with five forecast buckets, two position states, and three time-of-session buckets. The script selects the policy by validation net return, with cash as the fallback.
6. **Simulate trades.** Compare the frozen RL policy with a long-or-flat LSTM threshold policy and cash on the same test bars. The agent can buy, hold, or sell with 1× exposure. Entry and exit each cost half a basis point. The simulator liquidates at the last available close before a gap or session boundary, using the historical availability schedule. It assumes execution at the decision bar's close.

The trading report shows net return, fees per 10,000 units of starting capital, entries, exits, exposure, and drawdown. See the [loader guide](docs/DATA_LOADERS.md) for window eligibility and [rl_model.py](rl_model.py) for the environment and agent.

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (
    TimeSeriesWindowDataset,
    create_time_series_dataloaders,
    train_torch_model_dataloader,
    evaluate_torch_model_dataloader,
    load_fred_data,
    prepare_data,
)


def test_time_series_window_dataset_basic():
    print("Testing TimeSeriesWindowDataset basic functionality...")
    N = 100
    features = np.arange(N).reshape(-1, 1).astype(np.float32)
    targets = (np.arange(N) * 10).astype(np.float32)

    # 1. window_size=10, offset=0, target_size=1, stride=1
    ds = TimeSeriesWindowDataset(features, targets, window_size=10, offset=0, target_size=1, stride=1)
    # Target starts at index 10 for sample 0, last sample starts at index 89 with target at 99 -> 90 samples
    assert len(ds) == 90, f"Expected 90 samples, got {len(ds)}"
    x0, y0 = ds[0]
    assert x0.shape == (10, 1), f"Expected x0 shape (10, 1), got {x0.shape}"
    assert np.allclose(x0.numpy().flatten(), np.arange(0, 10)), "x0 values mismatch"
    assert np.allclose(y0.numpy().item(), 100.0), f"Expected y0=100, got {y0.numpy().item()}"

    x_last, y_last = ds[len(ds) - 1]
    assert np.allclose(x_last.numpy().flatten(), np.arange(89, 99)), "x_last values mismatch"
    assert np.allclose(y_last.numpy().item(), 990.0), f"Expected y_last=990, got {y_last.numpy().item()}"

    # 2. window_size=10, offset=5 (5 days gap), target_size=1, stride=1
    ds_offset = TimeSeriesWindowDataset(features, targets, window_size=10, offset=5, target_size=1, stride=1)
    # Target starts at index 15 for sample 0 -> 100 - 15 = 85 samples
    assert len(ds_offset) == 85, f"Expected 85 samples, got {len(ds_offset)}"
    x0, y0 = ds_offset[0]
    assert np.allclose(y0.numpy().item(), 150.0), f"Expected y0=150 with offset=5, got {y0.numpy().item()}"

    # 3. window_size=10, offset=0, target_size=3 (multi-step), stride=2
    ds_multi = TimeSeriesWindowDataset(features, targets, window_size=10, offset=0, target_size=3, stride=2)
    x0, y0 = ds_multi[0]
    assert y0.shape == (3,), f"Expected y0 shape (3,), got {y0.shape}"
    assert np.allclose(y0.numpy(), [100.0, 110.0, 120.0]), "y0 multi-step values mismatch"

    print("[PASS] TimeSeriesWindowDataset unit tests passed.")


def test_create_time_series_dataloaders_real_data():
    print("Testing create_time_series_dataloaders with real FRED data...")
    raw_df = prepare_data(load_fred_data("FRED"))
    print(f"Loaded FRED dataframe shape: {raw_df.shape}")

    loaders = create_time_series_dataloaders(
        data=raw_df,
        target_column="INR",
        feature_columns=["JPY", "EUR", "FFER", "US10YY"],
        window_size=30,
        offset=0,
        target_size=1,
        stride=1,
        batch_size=64,
        train_fraction=0.8,
        validation_fraction=0.1,
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    assert train_loader is not None, "train_loader is None"
    assert val_loader is not None, "val_loader is None"
    assert test_loader is not None, "test_loader is None"

    bx, by = next(iter(train_loader))
    print(f"Train batch - X shape: {bx.shape}, y shape: {by.shape}")
    assert bx.shape == (64, 30, 4), f"Expected (64, 30, 4), got {bx.shape}"
    assert by.shape == (64,), f"Expected (64,), got {by.shape}"

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print("[PASS] create_time_series_dataloaders passed.")
    return loaders


def test_training_and_evaluation_pipeline(loaders):
    print("Testing train_torch_model_dataloader and evaluate_torch_model_dataloader...")

    class SimpleLSTM(nn.Module):
        def __init__(self, in_features, hidden_dim=32):
            super().__init__()
            self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model = SimpleLSTM(in_features=len(loaders["feature_columns"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    history = train_torch_model_dataloader(
        model=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        epochs=5,
        log_every=1,
        early_stopping_patience=3,
    )

    assert len(history["train_losses"]) == 5
    assert len(history["val_losses"]) == 5

    eval_res = evaluate_torch_model_dataloader(
        model=model,
        dataloader=loaders["test_loader"],
        target_scaler=loaders["target_scaler"],
        plot=False,
    )

    assert "predictions" in eval_res
    assert "targets" in eval_res
    assert "metrics" in eval_res
    assert "directional_accuracy" in eval_res["metrics"]
    assert "mae" in eval_res["metrics"]
    assert "rmse" in eval_res["metrics"]

    print(f"Test Evaluation Directional Accuracy: {eval_res['metrics']['directional_accuracy']:.2f}%")
    print(f"Test MAE: {eval_res['metrics']['mae']:.4f} | RMSE: {eval_res['metrics']['rmse']:.4f}")
    print("[PASS] End-to-end training and evaluation test passed.")


if __name__ == "__main__":
    test_time_series_window_dataset_basic()
    loaders = test_create_time_series_dataloaders_real_data()
    test_training_and_evaluation_pipeline(loaders)
    print("\nALL DATALOADER TESTS COMPLETED SUCCESSFULLY!")

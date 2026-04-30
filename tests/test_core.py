import numpy as np
import pandas as pd

from src.train_classification import time_split, build_baseline_model
from src.train_cross_asset_model import prepare_target_dataset


def test_time_split_preserves_time_order():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "value": range(100),
        }
    )

    train_df, val_df, test_df = time_split(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_prepare_target_dataset_creates_binary_target():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "AAPL": [100, 101, 99, 102, 103],
            "MSFT": [50, 51, 52, 53, 54],
            "GOOGL": [70, 71, 72, 73, 74],
            "NVDA": [30, 31, 32, 33, 34],
        }
    )

    result = prepare_target_dataset(df, "AAPL", ["MSFT", "GOOGL", "NVDA"])

    assert "target_direction" in result.columns
    assert set(result["target_direction"].unique()).issubset({0, 1})
    assert len(result) == 5

def test_baseline_model_can_fit_and_predict():
    X = pd.DataFrame(
        {
            "ret_1d": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03],
            "ret_3d": [0.02, -0.01, 0.04, -0.02, 0.03, -0.04],
            "ret_5d": [0.03, -0.03, 0.05, -0.01, 0.04, -0.05],
            "ma_5": [100, 101, 102, 103, 104, 105],
            "ma_10": [99, 100, 101, 102, 103, 104],
            "vol_5d": [0.1, 0.2, 0.15, 0.18, 0.12, 0.22],
            "volume_change_1d": [0.05, -0.03, 0.02, -0.01, 0.04, -0.02],
            "spy_ret_1d": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01],
            "spy_ret_5d": [0.03, -0.02, 0.04, -0.03, 0.02, -0.04],
        }
    )

    y = np.array([1, 0, 1, 0, 1, 0])

    model = build_baseline_model()
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probs.shape == (len(y), 2)
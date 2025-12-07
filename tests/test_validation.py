import numpy as np
from src.data.generate import generate_weekly_data
from src.validation import rolling_origin_cv, mae, mape, r_squared


def test_mae_perfect():
    """MAE should be 0 for perfect predictions"""
    y = [100, 200, 300]
    assert mae(y, y) == 0


def test_mae_known():
    """MAE calculation checked against known values"""
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]  # errors: 10, 10, 10
    assert mae(y_true, y_pred) == 10


def test_mape_known():
    """MAPE should give percentage error"""
    y_true = [100, 200]
    y_pred = [90, 180]  # 10% off each
    assert abs(mape(y_true, y_pred) - 0.10) < 0.001


def test_r_squared_perfect():
    """R^2 = 1 for perfect fit"""
    y = [1, 2, 3, 4, 5]
    assert r_squared(y, y) == 1.0


def test_r_squared_mean():
    """R^2 = 0 when predicting the mean."""
    y_true = [1, 2, 3, 4, 5]
    y_pred = [3, 3, 3, 3, 3]  # always predict mean
    assert abs(r_squared(y_true, y_pred)) < 0.001


def test_rolling_cv_runs():
    """Rolling CV should complete without errors."""
    df = generate_weekly_data(n_weeks=80, seed=456)
    results = rolling_origin_cv(
        df,
        min_train_weeks=52,
        test_weeks=4,
        step=8,
    )
    assert results["n_folds"] > 0
    assert "avg_mae" in results


def test_rolling_cv_respects_time():
    """Each fold should train on earlier data than it tests."""
    df = generate_weekly_data(n_weeks=80, seed=456)
    results = rolling_origin_cv(
        df,
        min_train_weeks=52,
        test_weeks=4,
        step=8,
    )
    for fold in results["folds"]:
        assert fold["train_weeks"] <= fold["test_start"]

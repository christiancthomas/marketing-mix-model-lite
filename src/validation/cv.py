"""
Rolling-origin cross-validation for time series:

- Standard k-fold CV doesn't work for time series because it leaks future data.
- Rolling origin respects temporal ordering: train on past, test on future.
"""

import numpy as np
from src.model import MMM
from src.validation.metrics import mae, mape, r_squared


def rolling_origin_cv(
    df,
    min_train_weeks=52,
    test_weeks=4,
    step=4,
    target_col="sales",
    **mmm_kwargs,
):
    """
    This rolling-origin CV for MMM evaluation starts with min_train_weeks
    of training data, predicts test_weeks ahead, then rolls forward by
    step weeks and repeats.

    Arguments:
        df: DataFrame with week column and spend/sales data
        min_train_weeks: minimum weeks to train on before first test
        test_weeks: how many weeks to predict at each fold
        step: how far to roll forward between folds
        target_col: column to predict
        **mmm_kwargs: passed to MMM constructor

    Returns:
        dict with fold-level and aggregate metrics
    """
    n = len(df)
    results = []

    fold = 0
    train_end = min_train_weeks

    while train_end + test_weeks <= n:
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end : train_end + test_weeks].copy()

        # Fit on train, predict on test
        model = MMM(**mmm_kwargs)
        model.fit(train_df, target_col=target_col)
        preds = model.predict(test_df)
        actuals = test_df[target_col].values

        fold_result = {
            "fold": fold,
            "train_weeks": train_end,
            "test_start": train_end,
            "test_end": train_end + test_weeks,
            "mae": mae(actuals, preds),
            "mape": mape(actuals, preds),
            "r2": r_squared(actuals, preds),
        }
        results.append(fold_result)

        fold += 1
        train_end += step

    # Aggregate across folds
    if results:
        avg_mae = np.mean([r["mae"] for r in results])
        avg_mape = np.mean([r["mape"] for r in results])
        avg_r2 = np.mean([r["r2"] for r in results])
    else:
        avg_mae = avg_mape = avg_r2 = np.nan

    return {
        "folds": results,
        "n_folds": len(results),
        "avg_mae": avg_mae,
        "avg_mape": avg_mape,
        "avg_r2": avg_r2,
    }

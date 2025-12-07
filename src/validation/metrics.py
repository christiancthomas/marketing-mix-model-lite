"""
Error metrics for model evaluation.
"""

import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error: average magnitude of errors"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    - Returns as decimal (0.05 = 5% error).
    - Skips zeros in y_true to avoid division issues.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # avoid divide by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def r_squared(y_true, y_pred):
    """
    R^2: proportion of variance explained.
    1.0 = perfect, 0.0 = no better than mean, negative = worse than mean.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)

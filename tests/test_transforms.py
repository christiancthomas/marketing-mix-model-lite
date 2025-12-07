"""Tests for adstock and saturation transforms."""

import numpy as np
from src.transforms import adstock, saturation


def test_adstock_basic():
    """Just making sure it doesn't break"""
    result = adstock([100, 0, 0], 0.5)
    assert result[1] > 0  # should have carryover


def test_adstock_no_decay():
    """With decay=0, no carryover should happen."""
    result = adstock([100, 50, 25], 0.0)
    np.testing.assert_array_equal(result, [100, 50, 25])


def test_adstock_full_decay():
    """everything accumulates with decay=1"""
    result = adstock([100, 100, 100], 1.0)
    assert result[2] == 300  # 100 + 100 + 100


def test_adstock_decay_shape():
    """Carryover should decay geometrically."""
    result = adstock([100, 0, 0, 0], 0.5)
    # Week 0: 100, Week 1: 50, Week 2: 25, Week 3: 12.5
    np.testing.assert_array_almost_equal(result, [100, 50, 25, 12.5])


def test_saturation_sqrt():
    """sqrt should dampen large values more than small."""
    result = saturation([0, 1, 4, 100], method="sqrt")
    np.testing.assert_array_almost_equal(result, [0, 1, 2, 10])


def test_saturation_log():
    """log1p should handle zero gracefully."""
    result = saturation([0, 1, 100], method="log")
    assert result[0] == 0  # log(1+0) = 0
    assert result[1] > 0
    assert result[2] > result[1]  # monotone increasing


def test_saturation_monotone():
    """Saturation should always be monotone increasing."""
    x = np.linspace(0, 1000, 100)
    for method in ["sqrt", "log"]:
        result = saturation(x, method=method)
        diffs = np.diff(result)
        assert np.all(diffs >= 0), f"{method} not monotone"

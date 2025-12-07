"""
Adstock transformation for marketing spend.

- Captures how ad effects decay over time
  - a $10k spend in week 1 still influences
    week 2, 3, etc. with diminishing impact.
"""

import numpy as np


def adstock(x, decay_rate):
    """
    Apply geometric adstock decay to a spend series.
    - x: array of spend values (one per time period)
    - decay_rate: retention rate per period (0-1). Higher = longer carryover.
        e.g., 0.7 means 70% of effect carries to next period.

    Returns a transformed array with carryover effects.

    Example:
        spend = [100, 0, 0, 0]
        adstock(spend, 0.5) -> [100, 50, 25, 12.5]
    """
    x = np.asarray(x, dtype=float)

    if not 0 <= decay_rate <= 1:
        raise ValueError("decay_rate must be between 0 and 1")

    result = np.zeros_like(x)
    result[0] = x[0]

    for t in range(1, len(x)):
        result[t] = x[t] + decay_rate * result[t - 1]

    return result

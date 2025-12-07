"""
Saturation transformation for marketing spend.

Models diminishing returns - doubling spend doesn't double impact.
"""

import numpy as np


def saturation(x, method="sqrt"):
    """
    Apply saturation curve to capture diminishing returns.
    Takes as arguments, x: array of spend values
    and method: transformation type
        - "sqrt": square root (moderate saturation)
        - "log": natural log (aggressive saturation, requires x > 0)

    Returns a transformed array with diminishing returns applied
    """
    x = np.asarray(x, dtype=float)

    if method == "sqrt":
        return np.sqrt(np.maximum(x, 0))
    elif method == "log":
        # log1p handles x=0 gracefully: log(1+x)
        return np.log1p(np.maximum(x, 0))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sqrt' or 'log'.")

"""
Sales decomposition: break predicted sales into channel contributions.

The model's coefficients are in scaled space, so we need to unscale
them to get real dollar contributions per channel.
"""

import numpy as np
import pandas as pd


def decompose_sales(model, df):
    """
    Break down predicted sales into base + channel + seasonality contributions.

    Returns DataFrame with columns:
    - base: intercept (organic sales baseline)
    - {channel}: contribution from each marketing channel
    - seasonality: combined fourier term effects
    - predicted: total (should match model.predict)
    """
    if model.model is None:
        raise ValueError("Model not fitted yet")

    # Build features the same way the model does
    X = model._build_features(df)

    # Get coefficients in original (unscaled) space
    # scaled prediction: y = X_scaled @ coef + intercept
    # where X_scaled = (X - mean) / scale
    # so: y = (X - mean) / scale @ coef + intercept
    #       = X @ (coef / scale) - mean @ (coef / scale) + intercept
    # The per-feature contribution is: X[feature] * (coef / scale)
    # The intercept absorbs the mean correction

    coefs_scaled = model.model.coef_
    scale = model.scaler.scale_

    # Unscale coefficients
    coefs_unscaled = coefs_scaled / scale

    result = pd.DataFrame(index=df.index)

    # Base = intercept + mean correction term
    mean_correction = np.sum(model.scaler.mean_ * coefs_scaled / scale)
    result["base"] = model.model.intercept_ - mean_correction

    # Channel contributions
    for col in model.spend_cols_:
        channel = col.replace("spend_", "")
        feature_name = f"{channel}_transformed"
        idx = model.feature_names_.index(feature_name)
        result[channel] = X[feature_name].values * coefs_unscaled[idx]

    # Seasonality (combine all fourier terms)
    seasonality = np.zeros(len(df))
    for feature_name in model.feature_names_:
        if feature_name.startswith("sin_") or feature_name.startswith("cos_"):
            idx = model.feature_names_.index(feature_name)
            seasonality += X[feature_name].values * coefs_unscaled[idx]
    result["seasonality"] = seasonality

    # Control variables if present
    for control in ["promo", "competitor_launch"]:
        if control in model.feature_names_:
            idx = model.feature_names_.index(control)
            result[control] = X[control].values * coefs_unscaled[idx]

    # Total should match model.predict()
    result["predicted"] = result.drop(columns=["predicted"], errors="ignore").sum(axis=1)

    return result


def contribution_summary(model, df):
    """
    Aggregate contribution by channel over the full period.

    Returns dict with total contribution per channel and percentages.
    """
    decomp = decompose_sales(model, df)

    # Get channel columns (exclude base, seasonality, predicted, controls)
    exclude = {"base", "seasonality", "predicted", "promo", "competitor_launch"}
    channels = [c for c in decomp.columns if c not in exclude]

    total_sales = decomp["predicted"].sum()

    summary = {
        "total_sales": total_sales,
        "base": decomp["base"].sum(),
        "base_pct": decomp["base"].sum() / total_sales,
        "seasonality": decomp["seasonality"].sum(),
        "channels": {},
    }

    for channel in channels:
        contrib = decomp[channel].sum()
        summary["channels"][channel] = {
            "contribution": contrib,
            "pct_of_total": contrib / total_sales,
        }

    return summary

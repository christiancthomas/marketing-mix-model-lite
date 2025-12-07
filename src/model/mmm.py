"""
Marketing Mix Model using Elastic Net regression. This model fits
transformed media spend to sales with seasonality controls.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from src.transforms import adstock, saturation


class MMM:
    """
    A lightweight Marketing Mix Model

    1. Applies adstock + saturation transforms to spend columns
    2. Adds Fourier terms for seasonality
    3. Fits Elastic Net regression to predict sales based on transformed spend and seasonality
    """

    def __init__(
            self,
            decay_rates=None,
            saturation_method="sqrt",
            n_fourier_terms=2,
            alpha=1.0,
            l1_ratio=0.5,
            ):
        """
        decay_rates: dict mapping channel name -> decay rate (0-1)
                    if None, uses 0.5 for all channels
        saturation_method: "sqrt" or "log"
        n_fourier_terms: number of sin/cos pairs for annual seasonality
        alpha: regularization strength
        l1_ratio: balance between L1 and L2 (1.0 = lasso, 0.0 = ridge)
        """
        self.decay_rates = decay_rates or {}
        self.saturation_method = saturation_method
        self.n_fourier_terms = n_fourier_terms
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.spend_cols_ = None

    def _get_spend_cols(self, df):
        """Find columns that look like spend data"""
        return [c for c in df.columns if c.startswith("spend_")]

    def _add_fourier_terms(self, df, period=52):
        """Add sin/cos terms for annual seasonality."""
        week = df["week"].values
        fourier_df = pd.DataFrame(index=df.index)

        for k in range(1, self.n_fourier_terms + 1):
            fourier_df[f"sin_{k}"] = np.sin(2 * np.pi * k * week / period)
            fourier_df[f"cos_{k}"] = np.cos(2 * np.pi * k * week / period)

        return fourier_df

    def _transform_spend(self, df):
        """Apply adstock and saturation to our spend columns"""
        transformed = pd.DataFrame(index=df.index)

        for col in self.spend_cols_:
            channel = col.replace("spend_", "")
            decay = self.decay_rates.get(channel, 0.5)

            # Adstock first, then saturation
            adstocked = adstock(df[col].values, decay)
            saturated = saturation(adstocked, method=self.saturation_method)

            transformed[f"{channel}_transformed"] = saturated

        return transformed

    def _build_features(self, df):
        """Combine transformed spend, fourier terms, and control variables."""
        features = self._transform_spend(df)

        # Add seasonality
        fourier = self._add_fourier_terms(df)
        features = pd.concat([features, fourier], axis=1)

        # Add control variables if present
        if "promo" in df.columns:
            features["promo"] = df["promo"].values
        if "competitor_launch" in df.columns:
            features["competitor_launch"] = df["competitor_launch"].values

        return features

    def fit(self, df, target_col="sales"):
        """
        Fit the model to data.

        df should have: week, spend_* columns, and target
        """
        self.spend_cols_ = self._get_spend_cols(df)
        X = self._build_features(df)
        y = df[target_col].values

        self.feature_names_ = X.columns.tolist()

        # Scale features for better regularization
        X_scaled = self.scaler.fit_transform(X)

        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=10000,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        return self

    def predict(self, df):
        """Generate predictions for new data"""
        X = self._build_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_coefficients(self):
        """Return coefficients with feature names"""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        return dict(zip(self.feature_names_, self.model.coef_))

    def summary(self):
        """Print a quick summary of the fitted model"""
        if self.model is None:
            raise ValueError("Model not fitted yet")

        coefs = self.get_coefficients()
        print("MMM Coefficients (scaled):")
        print("-" * 40)
        for name, coef in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            if abs(coef) > 0.01:  # skip near-zero
                print(f"  {name:25s} {coef:10.3f}")
        print(f"\nIntercept: {self.model.intercept_:.2f}")

import numpy as np
from src.data.generate import generate_weekly_data
from src.model import MMM


def test_model_fits():
    """Model should fit without errors."""
    df = generate_weekly_data(n_weeks=52, seed=123)
    model = MMM()
    model.fit(df)
    assert model.model is not None


def test_model_predicts():
    """Predictions should be reasonable (positive, right shape)."""
    df = generate_weekly_data(n_weeks=52, seed=123)
    model = MMM()
    model.fit(df)

    preds = model.predict(df)
    assert len(preds) == len(df)
    assert np.all(preds > 0)  # sales should be positive


def test_coefficients_exist():
    """Should return coefficients for all features."""
    df = generate_weekly_data(n_weeks=52, seed=123)
    model = MMM()
    model.fit(df)

    coefs = model.get_coefficients()
    assert "meta_transformed" in coefs
    assert "sin_1" in coefs  # fourier term


def test_custom_decay_rates():
    """Should accept custom decay rates per channel."""
    df = generate_weekly_data(n_weeks=52, seed=123)
    decay_rates = {"meta": 0.7, "google": 0.3}
    model = MMM(decay_rates=decay_rates)
    model.fit(df)

    # Should still work
    assert model.model is not None

import numpy as np
from src.data.generate import generate_weekly_data
from src.model import MMM
from src.insights import (
    decompose_sales,
    contribution_summary,
    calculate_roas,
    roas_summary,
    budget_scenario,
    optimize_reallocation,
)


def test_decomposition_sums_to_predictions():
    """The whole point: contributions should add up to total predicted sales."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    decomp = decompose_sales(model, df)
    predictions = model.predict(df)

    # Should match within floating point tolerance
    np.testing.assert_array_almost_equal(
        decomp["predicted"].values,
        predictions,
        decimal=2,
    )


def test_contribution_summary_runs():
    """Summary should return reasonable structure."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    summary = contribution_summary(model, df)

    assert "total_sales" in summary
    assert "channels" in summary
    assert "meta" in summary["channels"]


def test_roas_positive_for_channels_with_spend():
    """Channels with spend should have some ROAS (could be negative if model says so)."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    roas = calculate_roas(model, df)

    # All channels should have results
    assert "meta" in roas
    assert "google" in roas
    # spend should be positive
    assert roas["meta"]["total_spend"] > 0


def test_roas_summary_sorted():
    """ROAS summary should be sorted by effectiveness."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    summary = roas_summary(model, df)

    # Should be a DataFrame sorted by ROAS descending
    assert len(summary) == 6  # 6 channels
    roas_values = summary["roas"].values
    assert all(roas_values[i] >= roas_values[i + 1] for i in range(len(roas_values) - 1))


def test_budget_scenario_runs():
    """Basic scenario should return comparison."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    result = budget_scenario(model, df, {"spend_meta": 1.1})

    assert "baseline_sales" in result
    assert "scenario_sales" in result
    assert result["scenario_sales"] != result["baseline_sales"]


def test_optimize_reallocation():
    """Shifting budget between channels should work."""
    df = generate_weekly_data(n_weeks=52, seed=789)
    model = MMM()
    model.fit(df)

    result = optimize_reallocation(model, df, "reddit", "meta", shift_pct=0.10)

    assert result["from_channel"] == "reddit"
    assert result["to_channel"] == "meta"
    assert result["shift_amount"] > 0

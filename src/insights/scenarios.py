"""
Budget reallocation scenarios to answer questions like:
- "What if we shifted 10% of Reddit spend to Meta?"
"""

import pandas as pd


def budget_scenario(model, df, reallocations):
    """
    Predict sales under a budget reallocation scenario.

    reallocations: dict mapping spend column -> multiplier
        e.g., {"spend_meta": 1.1, "spend_reddit": 0.9}
        means +10% to Meta, -10% from Reddit

    Returns dict with baseline vs scenario comparison
    """
    df_scenario = df.copy()

    # Apply budget changes
    spend_changes = {}
    for spend_col, multiplier in reallocations.items():
        if spend_col not in df.columns:
            raise ValueError(f"Column {spend_col} not found in data")

        original = df[spend_col].sum()
        df_scenario[spend_col] = df[spend_col] * multiplier
        new = df_scenario[spend_col].sum()
        spend_changes[spend_col] = {
            "original": original,
            "new": new,
            "delta": new - original,
        }

    # Get predictions
    baseline_sales = model.predict(df).sum()
    scenario_sales = model.predict(df_scenario).sum()

    return {
        "baseline_sales": baseline_sales,
        "scenario_sales": scenario_sales,
        "sales_delta": scenario_sales - baseline_sales,
        "sales_lift_pct": (scenario_sales - baseline_sales) / baseline_sales,
        "spend_changes": spend_changes,
    }


def optimize_reallocation(model, df, source_channel, target_channel, shift_pct=0.10):
    """
    Convenience function: shifts budget from one channel to another.

    Example: optimize_reallocation(model, df, "reddit", "meta", 0.10)
    shifts 10% of Reddit spend to Meta.
    """
    source_col = f"spend_{source_channel}"
    target_col = f"spend_{target_channel}"

    # Calculate how much to move
    source_spend = df[source_col].sum()
    shift_amount = source_spend * shift_pct

    # Figure out multipliers
    target_spend = df[target_col].sum()
    source_multiplier = 1 - shift_pct
    target_multiplier = 1 + (shift_amount / target_spend) if target_spend > 0 else 1

    result = budget_scenario(
        model, df,
        {source_col: source_multiplier, target_col: target_multiplier}
    )

    result["shift_amount"] = shift_amount
    result["from_channel"] = source_channel
    result["to_channel"] = target_channel

    return result

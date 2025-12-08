"""
ROAS (Return on Ad Spend) calculations

ROAS = incremental sales attributed to channel / spend on channel

Note: because of saturation, ROAS isn't constant — it decreases as spend
increases (diminishing returns). These functions give you the effective
ROAS over the observed data period.
"""

import pandas as pd
from src.insights.decompose import decompose_sales


def calculate_roas(model, df):
    """
    Calculates ROAS per channel:

    Returns dict with:
    - total_spend: sum of spend
    - total_contribution: sum of attributed sales
    - roas: contribution / spend
    """
    decomp = decompose_sales(model, df)

    results = {}
    for col in model.spend_cols_:
        channel = col.replace("spend_", "")
        total_spend = df[col].sum()
        total_contribution = decomp[channel].sum()

        results[channel] = {
            "total_spend": total_spend,
            "total_contribution": total_contribution,
            "roas": total_contribution / total_spend if total_spend > 0 else 0,
        }

    return results


def roas_summary(model, df):
    roas_data = calculate_roas(model, df)

    rows = []
    for channel, data in roas_data.items():
        rows.append({
            "channel": channel,
            "spend": data["total_spend"],
            "contribution": data["total_contribution"],
            "roas": data["roas"],
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("roas", ascending=False)
    return summary.reset_index(drop=True)

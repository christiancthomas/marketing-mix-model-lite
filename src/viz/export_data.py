import os
import pandas as pd
from src.data.generate import generate_weekly_data
from src.model import MMM
from src.insights import decompose_sales, roas_summary


def export_decomposition(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    # Generate data and fit model
    df = generate_weekly_data(n_weeks=104, seed=42)
    model = MMM()
    model.fit(df)

    # Get decomposition
    decomp = decompose_sales(model, df)
    decomp["week"] = df["week"].values

    # Reorder columns for nicer output
    cols = ["week", "base", "meta", "google", "tiktok", "reddit", "x", "twitch", "seasonality", "predicted"]
    decomp = decomp[[c for c in cols if c in decomp.columns]]

    decomp.to_csv(f"{output_dir}/decomposition.csv", index=False)
    print(f"Exported decomposition to {output_dir}/decomposition.csv")

    # Also export ROAS summary
    roas = roas_summary(model, df)
    roas.to_csv(f"{output_dir}/roas_summary.csv", index=False)
    print(f"Exported ROAS summary to {output_dir}/roas_summary.csv")

    return decomp, roas


if __name__ == "__main__":
    export_decomposition()

"""
Synthetic data generator for MMM.
Creates weekly spend/sales data with seasonality, promos, and other noise.
"""

import numpy as np
import pandas as pd


def generate_weekly_data(n_weeks=104, seed=42):
    """
    Generates synthetic marketing mix data

    Simulates ~2 years of weekly data for a game with:
    - 6 marketing channels (Meta, Google Ads, TikTok, Reddit, X, Twitch)
      - Seasonal patterns (holiday bumps, summer lulls)
      - Occasional promos and competitor launches
      - A launch spike in the first few weeks
    """
    rng = np.random.default_rng(seed)

    weeks = np.arange(n_weeks)
    start_date = pd.Timestamp("2023-01-01")
    dates = [start_date + pd.Timedelta(weeks=int(w)) for w in weeks]

    # Channel spend
    # Base spend patterns with some correlation (campaigns often run together)
    base_activity = rng.uniform(0.5, 1.5, n_weeks)  # shared marketing "intensity"

    # Major paid channels
    spend_meta = base_activity * rng.uniform(15000, 80000, n_weeks)
    spend_google = base_activity * rng.uniform(10000, 60000, n_weeks)
    spend_tiktok = base_activity * rng.uniform(5000, 40000, n_weeks)

    # Smaller/niche channels
    spend_reddit = base_activity * rng.uniform(2000, 15000, n_weeks)
    spend_x = base_activity * rng.uniform(3000, 20000, n_weeks)

    # Twitch = paid creator activations (influencer spend, not platform ads)
    spend_twitch = base_activity * rng.uniform(5000, 35000, n_weeks)

    # Launch spike: heavy spend in first 4 weeks
    launch_multiplier = np.array([3.0, 2.5, 1.8, 1.3] + [1.0] * (n_weeks - 4))
    spend_meta *= launch_multiplier
    spend_google *= launch_multiplier
    spend_tiktok *= launch_multiplier
    spend_reddit *= launch_multiplier
    spend_x *= launch_multiplier
    spend_twitch *= launch_multiplier

    # Events
    # Promos: ~8% of weeks have a sale/promo
    promo = rng.random(n_weeks) < 0.08

    # Competitor launches: ~5% of weeks, tends to hurt sales
    competitor_launch = rng.random(n_weeks) < 0.05

    # Seasonality
    # Annual cycle: peak around holidays (week ~48-52), dip in summer
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * (weeks - 48) / 52)

    # Sales generation
    base_sales = 50000

    # Channel contributions (these are the "true" effects we'll try to recover)
    # Diminishing returns baked in via sqrt
    # Coefficients reflect relative effectiveness: Meta/Google strongest, niche channels weaker
    channel_effect = (
        1.2 * np.sqrt(spend_meta) +
        1.0 * np.sqrt(spend_google) +
        0.7 * np.sqrt(spend_tiktok) +
        0.3 * np.sqrt(spend_reddit) +
        0.2 * np.sqrt(spend_x) +
        0.5 * np.sqrt(spend_twitch)
    )

    # Launch spike for sales too
    launch_sales_boost = np.array([80000, 50000, 30000, 15000] + [0] * (n_weeks - 4))

    # Promo lift: +20% when active
    promo_lift = np.where(promo, 0.20, 0.0)

    # Competitor hit: -10% when they launch
    competitor_hit = np.where(competitor_launch, -0.10, 0.0)

    # Combine everything
    sales = (
        (base_sales + channel_effect + launch_sales_boost) *
        seasonality *
        (1 + promo_lift + competitor_hit)
    )

    # Add noise (~5% CV)
    noise = rng.normal(1.0, 0.05, n_weeks)
    sales = sales * noise
    sales = np.maximum(sales, 0).round().astype(int)

    df = pd.DataFrame({
        "week": weeks,
        "date": dates,
        "spend_meta": spend_meta.round(2),
        "spend_google": spend_google.round(2),
        "spend_tiktok": spend_tiktok.round(2),
        "spend_reddit": spend_reddit.round(2),
        "spend_x": spend_x.round(2),
        "spend_twitch": spend_twitch.round(2),  # paid creator activations
        "promo": promo.astype(int),
        "competitor_launch": competitor_launch.astype(int),
        "sales": sales,
    })

    return df


if __name__ == "__main__":
    import os

    # Make sure data dir exists
    os.makedirs("data", exist_ok=True)

    df = generate_weekly_data()
    df.to_csv("data/weekly_data.csv", index=False)
    print(f"Generated {len(df)} weeks of data -> data/weekly_data.csv")

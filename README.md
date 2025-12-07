# Marketing Mix Model — Lite

Game publishers allocate budgets across channels like Twitch, YouTube, and Paid Social, but quantifying which channels drive incremental sales after the chaos of a launch remains difficult outside of user-level attribution. This lightweight Marketing Mix Model estimates channel contributions across aggregate conversion and spend data.

## Quick Start

```bash
git clone https://github.com/christiancthomas/marketing-mix-model-lite.git
cd marketing-mix-model-lite
```

*I will include setup instructions as features are added.*

## What's Here

- Synthetic data generation (*planned*)
- Adstock and saturation transforms (*planned*)
- Elastic Net regression (*planned*)
  - with seasonality (*TBD*)
- Channel contribution analysis (*planned*)

## Methodology

Marketing Mix Models decompose aggregate sales into base (organic) and incremental (media-driven) components. This implementation uses three core transforms standard in MMM literature.

### Adstock decay

In Marketing Mix Modeling, **Adstock** captures how marketing doesn't just influence a conversion in the moment. Instead, it builds awareness and influence over time, and decays post-exposure, resulting in a "lagging" effect.

Search ads might decay quickly (days) since intent is immediate, while brand channels like YouTube persist longer (weeks). The geometric adstock transformation is an industry standard, introduced in econometric marketing models by [Broadbent (1984)](https://www.warc.com/content/paywall/article/A1986_WARC_1539/the_phenomenon_of_adstock/en-GB) and formalized in most MMM frameworks.

### Saturation curves

**Saturation** models diminishing returns. This captures the phenomenon where doubling spend typically doesn't result in the doubling of incremental sales. This is especially true at higher spend. I'll apply this via log or square-root mathematical transforms to media variables before regression in an attempt to capture this effect.

Marginal ROAS decreases as spend increases due to features such as audience overlap, audience saturation, and creative fatigue. This non-linear relationship is fundamental to media planning and widely understood in the marketing domain and should be accounted for in MMM applications. The specific curve shape varies by channel. For example, smaller paid social channels (e.g., Reddit) may saturate faster than broader reach networks (Meta, Google, etc.).

### Regression model

**Elastic Net regression** combines L1 (lasso) and L2 (ridge) penalties for variable selection and coefficient stability. This model appears to handle seasonal effects which I need to capture annual patterns without overfitting.

**Why not standard regression?** In gaming, marketing channels are highly correlated. For example, YouTube & Google Search campaigns often run alongside Twitch sponsorships and paid social pushes to take advantage of specific marketing beats for moments like DLC launches or large updates. This multicollinearity makes ordinary least squares regression less stable. Small data changes have been known to produce wildly different attribution estimates.

In my research, Elastic Net addresses this effect through dual penalties. The L1 penalty (lasso) hanldes automatic variable selection, zeroing out less important channels. The L2 penalty (ridge) shrinks correlated coefficients toward each other, preventing any single channel from getting extreme attribution. It's my opinion that this combination yields more stable, and reasonable results for the gaming industry.

**Why this matters for MMM:** Limited time-series data is a common challenge for the application of MMM in gaming (we're lucky to get even a year's worth of weekly data/observations) relative to the number of marketing variables. In an ideal scenario, we could have ~100 weeks but potentially 10+ channels plus seasonality terms. We're often in a scenario where we have **more variables than we have reliable data points**. Without regularization, the model can perfectly fit the training data by memorizing noise, leading to poor predictions on new periods. This approach aligns with industry practices at [Google](https://research.google/pubs/pub46001/) and [Facebook's Robyn](https://github.com/facebookexperimental/Robyn) MMM frameworks.

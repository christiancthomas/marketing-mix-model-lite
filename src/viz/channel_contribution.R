# stacked area chart for hannel contribution over time
# Run from project root:
#   Rscript src/viz/channel_contribution.R

library(ggplot2)
library(tidyr)
library(dplyr)

# read decomposition data (run export_data.py first)
decomp <- read.csv("data/decomposition.csv")

# reshape for stacking
channels <- c("meta", "google", "tiktok", "reddit", "x", "twitch")
decomp_long <- decomp %>%
  select(week, all_of(channels)) %>%
  pivot_longer(cols = all_of(channels), names_to = "channel", values_to = "contribution")

# Order channels by total contribution (biggest on bottom)
channel_order <- decomp_long %>%
  group_by(channel) %>%
  summarise(total = sum(contribution)) %>%
  arrange(desc(total)) %>%
  pull(channel)

decomp_long$channel <- factor(decomp_long$channel, levels = rev(channel_order))

# color palette
channel_colors <- c(
  "google" = "#5555F2",  # Blurple
  "meta" = "#8A57F0",    # Purple
  "tiktok" = "#F55B53",  # Red
  "reddit" = "#F5CF22",  # Yellow
  "x" = "#2FF579",       # Green
  "twitch" = "#46F2FB"   # Cyan
)

# Stacked area chart
p <- ggplot(decomp_long, aes(x = week, y = contribution, fill = channel)) +
  geom_area(alpha = 0.8) +
  scale_fill_manual(values = channel_colors) +
  labs(
    title = "Channel Contribution Over Time",
    subtitle = "Incremental sales attributed to each marketing channel",
    x = "Week",
    y = "Sales Contribution ($)",
    fill = "Channel"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )

# Save to reports/figures
dir.create("reports/figures", recursive = TRUE, showWarnings = FALSE)
ggsave("reports/figures/channel_contribution.png", p, width = 10, height = 6, dpi = 150)

cat("Saved: reports/figures/channel_contribution.png\n")

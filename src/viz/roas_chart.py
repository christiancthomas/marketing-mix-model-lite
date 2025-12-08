import os
import matplotlib.pyplot as plt
import pandas as pd

# Brand palette
CHANNEL_COLORS = {
    "google": "#5555F2",  # Blurple
    "meta": "#8A57F0",    # Purple
    "tiktok": "#F55B53",  # Red
    "reddit": "#F5CF22",  # Yellow
    "x": "#2FF579",       # Green
    "twitch": "#46F2FB",  # Cyan
}


def plot_roas(input_file="data/roas_summary.csv", output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    roas = pd.read_csv(input_file)

    # sort by ROAS descending (should already be sorted, but just in case)
    roas = roas.sort_values("roas", ascending=True)  # ascending for horizontal bars

    # Use brand colors per channel
    colors = [CHANNEL_COLORS.get(ch, "#5555F2") for ch in roas["channel"]]
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(roas["channel"], roas["roas"], color=colors, edgecolor="white")

    ax.set_xlabel("ROAS ($ return per $ spent)")
    ax.set_title("Return on Ad Spend by Channel", fontweight="bold")
    ax.axvline(x=0, color="#D1D5DB", linewidth=0.5)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, roas["roas"]):
        ax.text(
            val + 0.01 if val >= 0 else val - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/roas_by_channel.png", dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/roas_by_channel.png")


if __name__ == "__main__":
    plot_roas()

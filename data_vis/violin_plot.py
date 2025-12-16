import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL STYLE (match your existing figures, but readable)
# =============================================================================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 20

# =============================================================================
# PATHS (unchanged)
# =============================================================================
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures_more")
FIGURES_DIR.mkdir(exist_ok=True)


def save_figure(fig, name):
    """Save figure in both PDF and PNG formats (overwrite existing)."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Overwritten: {name}.pdf / {name}.png")


# =============================================================================
# READABLE VIOLIN PLOT (ONLY FIGURE C)
# =============================================================================
def generate_score_distribution_comparison():
    print("\n[Figure C] Regenerating score distribution violin plot (readable version)...")

    # Load distribution statistics
    with open(RESULTS_DIR / "score_distribution_analysis.json") as f:
        dist_data = json.load(f)

    model_order = [
        "Custom_ViT_FineTuned",
        "ResNet50_Pretrained",
        "Base_ViT_Pretrained",
    ]

    model_names = [
        "Custom ViT\nFine-tuned",
        "ResNet-50\nPretrained",
        "Base ViT\nPretrained",
    ]

    # -------------------------------------------------------------------------
    # Load scores
    # -------------------------------------------------------------------------
    all_scores = []
    all_labels = []

    for model_key, display_name in zip(model_order, model_names):
        df = pd.read_csv(RESULTS_DIR / model_key / "score_distributions.csv")

        live_scores = df[df["label"] == "live"]["score"].values
        spoof_scores = df[df["label"] == "spoof"]["score"].values

        all_scores.extend(live_scores)
        all_labels.extend([f"{display_name}\nLive"] * len(live_scores))

        all_scores.extend(spoof_scores)
        all_labels.extend([f"{display_name}\nSpoof"] * len(spoof_scores))

    df_plot = pd.DataFrame({
        "Score": all_scores,
        "Model": all_labels
    })

    ordered_labels = []
    for name in model_names:
        ordered_labels.append(f"{name}\nLive")
        ordered_labels.append(f"{name}\nSpoof")

    # -------------------------------------------------------------------------
    # FIGURE (same size as before)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    parts = ax.violinplot(
        [df_plot[df_plot["Model"] == lbl]["Score"].values for lbl in ordered_labels],
        showmeans=True,
        showmedians=True,
        widths=0.75
    )

    # Colors (unchanged palette, higher contrast)
    colors = [
        "#2E86AB", "#E63946",
        "#A23B72", "#F77F00",
        "#95A5A6", "#E74C3C"
    ]

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
        body.set_linewidth(1.2)

    # Mean / median styling (thicker)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2.8)

    parts["cmedians"].set_color("darkred")
    parts["cmedians"].set_linewidth(2.8)

    # -------------------------------------------------------------------------
    # AXES & LABELS (KEY READABILITY FIX)
    # -------------------------------------------------------------------------
    ax.set_xticks(range(1, len(ordered_labels) + 1))
    ax.set_xticklabels(
        ordered_labels,
        rotation=30,
        ha="right",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel("Prediction Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Score Distribution Comparison Across Models",
        fontsize=15,
        fontweight="bold",
        pad=14
    )

    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # MEAN ANNOTATIONS (BIGGER & CLEAR)
    # -------------------------------------------------------------------------
    for i, model_key in enumerate(model_order):
        stats = dist_data[model_key]

        live_mean = stats["live_scores"]["mean"]
        spoof_mean = stats["spoof_scores"]["mean"]

        ax.text(
            i * 2 + 1,
            live_mean,
            f"μ={live_mean:.3f}",
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9)
        )

        ax.text(
            i * 2 + 2,
            spoof_mean,
            f"μ={spoof_mean:.3f}",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9)
        )

    # -------------------------------------------------------------------------
    # LEGEND (READABLE)
    # -------------------------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="black", lw=3, label="Mean"),
        Line2D([0], [0], color="darkred", lw=3, label="Median"),
    ]

    ax.legend(
        handles=legend_items,
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    save_figure(fig, "score_distribution_comparison")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    generate_score_distribution_comparison()
    print("\n✔ Violin plot regenerated with improved readability")

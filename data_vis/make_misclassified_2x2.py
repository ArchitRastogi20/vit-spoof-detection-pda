import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# -----------------------------------------------------------------------------
# OPTIONAL: EXCLUSION LISTS (KEEP FOR SAFETY)
# -----------------------------------------------------------------------------
EXCLUDED_IMAGES = {
    "Custom_ViT_FineTuned": {
        "false_positives": set(),
        "false_negatives": set(),
    },
    "ResNet50_Pretrained": {
        "false_positives": set(),
        "false_negatives": set(),
    },
}

def filter_excluded(images, model, error_type):
    excluded = EXCLUDED_IMAGES.get(model, {}).get(error_type, set())
    return [img for img in images if img.name not in excluded]


# -----------------------------------------------------------------------------
# CORE FUNCTION
# -----------------------------------------------------------------------------
def generate_misclassified_2x2(model_name, output_name, title):
    print(f"[Figure] Generating 2x2 misclassified samples for {model_name}...")

    failed_dir = RESULTS_DIR / "failed_cases_analysis" / model_name
    fp_dir = failed_dir / "false_positives"
    fn_dir = failed_dir / "false_negatives"

    fp_all = sorted(fp_dir.glob("*.png"))
    fn_all = sorted(fn_dir.glob("*.png"))

    fp_images = filter_excluded(fp_all, model_name, "false_positives")[:2]
    fn_images = filter_excluded(fn_all, model_name, "false_negatives")[:2]

    if len(fp_images) < 2 or len(fn_images) < 2:
        raise RuntimeError("Need at least 2 FP and 2 FN images")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    # Clear axes
    for ax in axes.flatten():
        ax.axis("off")

    # -------------------------
    # False Positives (Top Row)
    # -------------------------
    for col in range(2):
        img_path = fp_images[col]
        score = float(img_path.stem.split("score")[1].split("_")[0])
        img = Image.open(img_path)

        axes[0, col].imshow(img)
        axes[0, col].set_title(
            f"FP: Score={score:.3f}\n(Spoof → Live)",
            fontsize=10,
            color="red",
        )
        axes[0, col].axis("off")

    # -------------------------
    # False Negatives (Bottom Row)
    # -------------------------
    for col in range(2):
        img_path = fn_images[col]
        score = float(img_path.stem.split("score")[1].split("_")[0])
        img = Image.open(img_path)

        axes[1, col].imshow(img)
        axes[1, col].set_title(
            f"FN: Score={score:.3f}\n(Live → Spoof)",
            fontsize=10,
            color="darkorange",
        )
        axes[1, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(FIGURES_DIR / f"{output_name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{output_name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved figures/{output_name}.pdf and .png")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    generate_misclassified_2x2(
        model_name="Custom_ViT_FineTuned",
        output_name="misclassified_vit",
        title="Custom ViT: Misclassified Samples",
    )

    generate_misclassified_2x2(
        model_name="ResNet50_Pretrained",
        output_name="misclassified_resnet",
        title="ResNet-50: Misclassified Samples",
    )

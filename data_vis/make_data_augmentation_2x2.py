import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SAMPLE_DIR = Path("sample_augmented_images")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

FIG_NAME = "data_augmentation"  # overwrite same figure

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11


# -----------------------------------------------------------------------------
# SAVE HELPER
# -----------------------------------------------------------------------------
def save_figure(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figures/{name}.pdf and .png")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def generate_data_augmentation_2x2():
    print("[Figure] Generating 2x2 data augmentation figure...")

    valid_samples = []

    for d in sorted(SAMPLE_DIR.iterdir()):
        if not d.is_dir():
            continue

        original = d / "spoof_original" / "original.jpg"
        augmented = d / "augmented" / "augmented.jpg"

        if original.exists() and augmented.exists():
            valid_samples.append(d)

    if len(valid_samples) < 2:
        raise RuntimeError("Need at least 2 valid samples for 2x2 grid")

    samples = valid_samples[:2]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    fig.suptitle(
        "Data Augmentation Examples\n(Original Top, Augmented Bottom)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    # Clear axes
    for ax in axes.flatten():
        ax.axis("off")

    for col, sample_dir in enumerate(samples):
        orig_path = sample_dir / "spoof_original" / "original.jpg"
        aug_path = sample_dir / "augmented" / "augmented.jpg"

        # Original
        img_orig = Image.open(orig_path)
        axes[0, col].imshow(img_orig)
        axes[0, col].set_title(f"Original {col+1}", fontsize=10)
        axes[0, col].axis("off")

        # Augmented
        img_aug = Image.open(aug_path)
        axes[1, col].imshow(img_aug)
        axes[1, col].set_title(f"Augmented {col+1}", fontsize=10)
        axes[1, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, FIG_NAME)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_data_augmentation_2x2()

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

FIG_NAME = "data_augmentation"  # SAME NAME as before

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10


# -----------------------------------------------------------------------------
# SAVE HELPER
# -----------------------------------------------------------------------------
def save_figure(fig, name):
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved {pdf_path}")
    print(f"[OK] Saved {png_path}")


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def generate_data_augmentation_figure():
    print("[Figure] Generating data augmentation grid...")

    valid_samples = []

    for d in sorted(SAMPLE_DIR.iterdir()):
        if not d.is_dir():
            continue

        original = d / "spoof_original" / "original.jpg"
        augmented = d / "augmented" / "augmented.jpg"

        if original.exists() and augmented.exists():
            valid_samples.append(d)

    if len(valid_samples) < 4:
        print(f"[WARN] Only {len(valid_samples)} valid samples found (need 4).")

    samples = valid_samples[:4]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(
        "Data Augmentation Examples (Original Top, Augmented Bottom)",
        fontsize=12,
        fontweight="bold",
    )

    # Initialize empty grid (prevents ghost axes)
    for ax in axes.flatten():
        ax.axis("off")

    for col, sample_dir in enumerate(samples):
        original_path = sample_dir / "spoof_original" / "original.jpg"
        augmented_path = sample_dir / "augmented" / "augmented.jpg"

        print(f"  Using sample: {sample_dir.name}")

        # Original
        img_orig = Image.open(original_path)
        axes[0, col].imshow(img_orig)
        axes[0, col].set_title(f"Original {col+1}", fontsize=9)
        axes[0, col].axis("off")

        # Augmented
        img_aug = Image.open(augmented_path)
        axes[1, col].imshow(img_aug)
        axes[1, col].set_title(f"Augmented {col+1}", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    save_figure(fig, FIG_NAME)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_data_augmentation_figure()

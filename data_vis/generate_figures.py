import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# GLOBAL IMAGE EXCLUSION LISTS (FOR REPORT SAFETY)
# =============================================================================

EXCLUDED_IMAGES = {
    "Custom_ViT_FineTuned": {
        "false_positives": set([
            "FP_score0.8125_513034.png"
        ]),
        "false_negatives": set([

        ])
    },
    "ResNet50_Pretrained": {
        "false_positives": set([
            "FP_score0.6770_500049.png",
            "FP_score0.6781_548494.png"
        ]),
        "false_negatives": set([
            # add here
        ])
    }
}


# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Paths
RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("GENERATING FIGURES FOR LATEX REPORT")
print("="*80)

def filter_excluded(images, model_name, error_type):
    """
    images: list[Path]
    model_name: str
    error_type: 'false_positives' | 'false_negatives'
    """
    excluded = EXCLUDED_IMAGES.get(model_name, {}).get(error_type, set())
    return [img for img in images if img.name not in excluded]


def save_figure(fig, name):
    """Save figure in both PDF and PNG formats"""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {name}.pdf and {name}.png")
    plt.close(fig)


# =============================================================================
# FIGURE 2: Data Augmentation Grid (2x4)
# =============================================================================
def generate_figure2():
    print("\n[Figure 2] Generating data augmentation grid...")
    
    sample_dir = Path('sample_augmented_images')
    
    # Get first 4 sample directories
    sample_dirs = sorted([d for d in sample_dir.iterdir() if d.is_dir()])[:4]
    
    if len(sample_dirs) < 4:
        print(f"Warning: Only found {len(sample_dirs)} samples, need 4")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Data Augmentation Examples (Original Top, Augmented Bottom)', 
                 fontsize=12, fontweight='bold')
    
    for col, sample_dir_path in enumerate(sample_dirs):
        original_path = sample_dir_path / 'spoof_original' / 'original.jpg'
        augmented_path = sample_dir_path / 'augmented' / 'augmented.jpg'
        
        print(f"  Sample {col+1}: {sample_dir_path.name}")
        print(f"    Original: {original_path}")
        print(f"    Augmented: {augmented_path}")
        
        if original_path.exists():
            img = Image.open(original_path)
            axes[0, col].imshow(img)
            axes[0, col].axis('off')
            axes[0, col].set_title(f'Original {col+1}', fontsize=9)
        
        if augmented_path.exists():
            img = Image.open(augmented_path)
            axes[1, col].imshow(img)
            axes[1, col].axis('off')
            axes[1, col].set_title(f'Augmented {col+1}', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'data_augmentation')


# =============================================================================
# FIGURE 4: ROC Curves Comparison
# =============================================================================
def generate_figure4():
    print("\n[Figure 4] Generating ROC curves comparison...")
    
    vit_roc = pd.read_csv(RESULTS_DIR / 'Custom_ViT_FineTuned' / 'roc_curve_data.csv')
    resnet_roc = pd.read_csv(RESULTS_DIR / 'ResNet50_Pretrained' / 'roc_curve_data.csv')
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot ROC curves
    ax.plot(vit_roc['fpr'], vit_roc['tpr'], 
            linewidth=2, label='Custom ViT (AUC=0.5665)', color='#2E86AB')
    ax.plot(resnet_roc['fpr'], resnet_roc['tpr'], 
            linewidth=2, label='ResNet-50 (AUC=0.5597)', color='#A23B72')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    save_figure(fig, 'roc_comparison')


# =============================================================================
# FIGURE 5: APCER vs BPCER for Custom ViT
# =============================================================================
def generate_figure5():
    print("\n[Figure 5] Generating APCER vs BPCER for Custom ViT...")
    
    df = pd.read_csv(RESULTS_DIR / 'Custom_ViT_FineTuned' / 'threshold_analysis.csv')
    
    # Use frr for APCER and far for BPCER (corrected)
    apcer = df['frr']
    bpcer = df['far']
    thresholds = df['threshold']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot trade-off curve
    ax.plot(apcer, bpcer, linewidth=2.5, color='#2E86AB', marker='o', 
            markersize=4, markevery=1)
    
    # Mark EER point
    eer_idx = df['threshold'].sub(0.5597).abs().idxmin()
    eer_apcer = apcer.iloc[eer_idx]
    eer_bpcer = bpcer.iloc[eer_idx]
    
    ax.scatter([eer_apcer], [eer_bpcer], color='red', s=150, 
               marker='*', zorder=5, label=f'EER=0.4526 (t=0.5597)')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('APCER (Attack Accepted as Genuine)', fontsize=11)
    ax.set_ylabel('BPCER (Genuine Rejected as Attack)', fontsize=11)
    ax.set_title('Custom ViT: APCER vs BPCER Trade-off', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    save_figure(fig, 'apcer_bpcer_vit')


# =============================================================================
# FIGURE 6: APCER vs BPCER for ResNet50
# =============================================================================
def generate_figure6():
    print("\n[Figure 6] Generating APCER vs BPCER for ResNet50...")
    
    df = pd.read_csv(RESULTS_DIR / 'ResNet50_Pretrained' / 'threshold_analysis.csv')
    
    # Use frr for APCER and far for BPCER (corrected)
    apcer = df['frr']
    bpcer = df['far']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot trade-off curve
    ax.plot(apcer, bpcer, linewidth=2.5, color='#A23B72', marker='s', 
            markersize=4, markevery=1)
    
    # Mark EER point
    eer_idx = df['threshold'].sub(0.5732).abs().idxmin()
    eer_apcer = apcer.iloc[eer_idx]
    eer_bpcer = bpcer.iloc[eer_idx]
    
    ax.scatter([eer_apcer], [eer_bpcer], color='red', s=150, 
               marker='*', zorder=5, label=f'EER=0.4405 (t=0.5732)')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('APCER (Attack Accepted as Genuine)', fontsize=11)
    ax.set_ylabel('BPCER (Genuine Rejected as Attack)', fontsize=11)
    ax.set_title('ResNet-50: APCER vs BPCER Trade-off', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    save_figure(fig, 'apcer_bpcer_resnet')


# =============================================================================
# FIGURE 7: Score Distribution for Custom ViT
# =============================================================================
def generate_figure7():
    print("\n[Figure 7] Generating score distribution for Custom ViT...")
    
    with open(RESULTS_DIR / 'score_distribution_analysis.json') as f:
        data = json.load(f)
    
    vit_data = data['Custom_ViT_FineTuned']
    
    df = pd.read_csv(RESULTS_DIR / 'Custom_ViT_FineTuned' / 'score_distributions.csv')
    live_scores = df[df['label'] == 'live']['score']
    spoof_scores = df[df['label'] == 'spoof']['score']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histograms
    ax.hist(live_scores, bins=50, alpha=0.6, color='#2E86AB', 
            label=f"Live (μ={vit_data['live_scores']['mean']:.3f}, σ={vit_data['live_scores']['std']:.3f})",
            edgecolor='black', linewidth=0.5)
    ax.hist(spoof_scores, bins=50, alpha=0.6, color='#E63946', 
            label=f"Spoof (μ={vit_data['spoof_scores']['mean']:.3f}, σ={vit_data['spoof_scores']['std']:.3f})",
            edgecolor='black', linewidth=0.5)
    
    # Vertical lines at means
    ax.axvline(vit_data['live_scores']['mean'], color='#2E86AB', 
               linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(vit_data['spoof_scores']['mean'], color='#E63946', 
               linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Prediction Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Custom ViT: Score Distribution (Live vs Spoof)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlim([0, 1])
    
    save_figure(fig, 'score_distribution_vit')


# =============================================================================
# FIGURE 8: Score Distribution for ResNet50
# =============================================================================
def generate_figure8():
    print("\n[Figure 8] Generating score distribution for ResNet50...")
    
    with open(RESULTS_DIR / 'score_distribution_analysis.json') as f:
        data = json.load(f)
    
    resnet_data = data['ResNet50_Pretrained']
    
    df = pd.read_csv(RESULTS_DIR / 'ResNet50_Pretrained' / 'score_distributions.csv')
    live_scores = df[df['label'] == 'live']['score']
    spoof_scores = df[df['label'] == 'spoof']['score']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histograms
    ax.hist(live_scores, bins=50, alpha=0.6, color='#2E86AB', 
            label=f"Live (μ={resnet_data['live_scores']['mean']:.3f}, σ={resnet_data['live_scores']['std']:.3f})",
            edgecolor='black', linewidth=0.5)
    ax.hist(spoof_scores, bins=50, alpha=0.6, color='#E63946', 
            label=f"Spoof (μ={resnet_data['spoof_scores']['mean']:.3f}, σ={resnet_data['spoof_scores']['std']:.3f})",
            edgecolor='black', linewidth=0.5)
    
    # Vertical lines at means
    ax.axvline(resnet_data['live_scores']['mean'], color='#2E86AB', 
               linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(resnet_data['spoof_scores']['mean'], color='#E63946', 
               linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Prediction Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('ResNet-50: Score Distribution (Live vs Spoof)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_xlim([0, 1])
    
    save_figure(fig, 'score_distribution_resnet')


# =============================================================================
# FIGURE 9: Confusion Matrix for Custom ViT at EER
# =============================================================================
def generate_figure9():
    print("\n[Figure 9] Generating confusion matrix for Custom ViT...")
    
    df = pd.read_csv(RESULTS_DIR / 'Custom_ViT_FineTuned' / 'threshold_analysis.csv')
    
    # Find EER threshold row
    eer_idx = df['threshold'].sub(0.5597).abs().idxmin()
    row = df.iloc[eer_idx]
    
    # Convert to integers
    cm = np.array([[int(row['tn']), int(row['fp'])], 
                   [int(row['fn']), int(row['tp'])]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                xticklabels=['Live', 'Spoof'],
                yticklabels=['Live', 'Spoof'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(f'Custom ViT: Confusion Matrix at EER (t=0.5597)', 
                 fontsize=12, fontweight='bold')
    
    save_figure(fig, 'confusion_matrix_vit')


# =============================================================================
# FIGURE 10: Confusion Matrix for ResNet50 at EER
# =============================================================================
def generate_figure10():
    print("\n[Figure 10] Generating confusion matrix for ResNet50...")
    
    df = pd.read_csv(RESULTS_DIR / 'ResNet50_Pretrained' / 'threshold_analysis.csv')
    
    # Find EER threshold row
    eer_idx = df['threshold'].sub(0.5732).abs().idxmin()
    row = df.iloc[eer_idx]
    
    # Convert to integers
    cm = np.array([[int(row['tn']), int(row['fp'])], 
                   [int(row['fn']), int(row['tp'])]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                cbar_kws={'label': 'Count'},
                xticklabels=['Live', 'Spoof'],
                yticklabels=['Live', 'Spoof'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title(f'ResNet-50: Confusion Matrix at EER (t=0.5732)', 
                 fontsize=12, fontweight='bold')
    
    save_figure(fig, 'confusion_matrix_resnet')


# =============================================================================
# FIGURE 11: Misclassified Samples for Custom ViT
# =============================================================================
def generate_figure11():
    print("\n[Figure 11] Generating misclassified samples for Custom ViT...")
    
    failed_dir = RESULTS_DIR / 'failed_cases_analysis' / 'Custom_ViT_FineTuned'
    fp_dir = failed_dir / 'false_positives'
    fn_dir = failed_dir / 'false_negatives'
    
    print(f"  FP directory: {fp_dir}")
    print(f"  FN directory: {fn_dir}")
    
    # Get 3 FP and 3 FN samples
    fp_images_all = sorted(fp_dir.glob('*.png'))
    fn_images_all = sorted(fn_dir.glob('*.png'))

    fp_images = filter_excluded(
        fp_images_all, "Custom_ViT_FineTuned", "false_positives"
    )[:3]

    fn_images = filter_excluded(
        fn_images_all, "Custom_ViT_FineTuned", "false_negatives"
    )[:3]

    
    print(f"  Found {len(fp_images)} FP images")
    print(f"  Found {len(fn_images)} FN images")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Custom ViT: Misclassified Samples', fontsize=14, fontweight='bold')
    
    # Top row: False Positives (Spoof predicted as Live)
    for col, img_path in enumerate(fp_images):
        score = float(img_path.stem.split('score')[1].split('_')[0])
        img = Image.open(img_path)
        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        axes[0, col].set_title(f'FP: Score={score:.3f}\n(Spoof→Live)', 
                               fontsize=9, color='red')
    
    # Bottom row: False Negatives (Live predicted as Spoof)
    for col, img_path in enumerate(fn_images):
        score = float(img_path.stem.split('score')[1].split('_')[0])
        img = Image.open(img_path)
        axes[1, col].imshow(img)
        axes[1, col].axis('off')
        axes[1, col].set_title(f'FN: Score={score:.3f}\n(Live→Spoof)', 
                               fontsize=9, color='orange')
    
    plt.tight_layout()
    save_figure(fig, 'misclassified_vit')


# =============================================================================
# FIGURE 12: Misclassified Samples for ResNet50
# =============================================================================
def generate_figure12():
    print("\n[Figure 12] Generating misclassified samples for ResNet50...")
    
    failed_dir = RESULTS_DIR / 'failed_cases_analysis' / 'ResNet50_Pretrained'
    fp_dir = failed_dir / 'false_positives'
    fn_dir = failed_dir / 'false_negatives'
    
    print(f"  FP directory: {fp_dir}")
    print(f"  FN directory: {fn_dir}")
    
    # Get 3 FP and 3 FN samples
    fp_images_all = sorted(fp_dir.glob('*.png'))
    fn_images_all = sorted(fn_dir.glob('*.png'))

    fp_images = filter_excluded(
        fp_images_all, "ResNet50_Pretrained", "false_positives"
    )[:3]

    fn_images = filter_excluded(
        fn_images_all, "ResNet50_Pretrained", "false_negatives"
    )[:3]

    
    print(f"  Found {len(fp_images)} FP images")
    print(f"  Found {len(fn_images)} FN images")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('ResNet-50: Misclassified Samples', fontsize=14, fontweight='bold')
    
    # Top row: False Positives (Spoof predicted as Live)
    for col, img_path in enumerate(fp_images):
        score = float(img_path.stem.split('score')[1].split('_')[0])
        img = Image.open(img_path)
        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        axes[0, col].set_title(f'FP: Score={score:.3f}\n(Spoof→Live)', 
                               fontsize=9, color='red')
    
    # Bottom row: False Negatives (Live predicted as Spoof)
    for col, img_path in enumerate(fn_images):
        score = float(img_path.stem.split('score')[1].split('_')[0])
        img = Image.open(img_path)
        axes[1, col].imshow(img)
        axes[1, col].axis('off')
        axes[1, col].set_title(f'FN: Score={score:.3f}\n(Live→Spoof)', 
                               fontsize=9, color='orange')
    
    plt.tight_layout()
    save_figure(fig, 'misclassified_resnet')


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    try:
        generate_figure2()   # Data augmentation
        generate_figure4()   # ROC comparison
        generate_figure5()   # APCER-BPCER ViT
        generate_figure6()   # APCER-BPCER ResNet
        generate_figure7()   # Score dist ViT
        generate_figure8()   # Score dist ResNet
        generate_figure9()   # Confusion matrix ViT
        generate_figure10()  # Confusion matrix ResNet
        generate_figure11()  # Misclassified ViT
        generate_figure12()  # Misclassified ResNet
        
        print("\n" + "="*80)
        print("ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"Output directory: {FIGURES_DIR.absolute()}")
        print(f"Generated {len(list(FIGURES_DIR.glob('*.pdf')))} PDF files")
        print(f"Generated {len(list(FIGURES_DIR.glob('*.png')))} PNG files")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
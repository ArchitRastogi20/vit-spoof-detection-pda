import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
FIGURES_DIR = Path('figures_more')
FIGURES_DIR.mkdir(exist_ok=True)

print("="*80)
print("GENERATING ADDITIONAL ANALYSIS FIGURES")
print("="*80)


def save_figure(fig, name):
    """Save figure in both PDF and PNG formats"""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {name}.pdf and {name}.png")
    plt.close(fig)


# =============================================================================
# FIGURE A: Fine-tuning Impact Bar Chart
# =============================================================================
def generate_finetuning_impact():
    print("\n[Figure A] Generating fine-tuning impact analysis...")
    
    # Load comparison data
    with open(RESULTS_DIR / 'model_comparison.json') as f:
        data = json.load(f)
    
    # Extract AUC values
    models_data = {item['model_name']: item for item in data['comparison_table']}
    
    base_vit_auc = models_data['Base_ViT_Pretrained']['roc_auc']
    custom_vit_auc = models_data['Custom_ViT_FineTuned']['roc_auc']
    
    improvement_pct = ((custom_vit_auc - base_vit_auc) / base_vit_auc) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Base ViT\n(No Fine-tuning)', 'Custom ViT\n(Fine-tuned)']
    auc_values = [base_vit_auc, custom_vit_auc]
    colors = ['#95A5A6', '#2E86AB']
    
    bars = ax.bar(models, auc_values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value labels on bars
    for i, (bar, auc) in enumerate(zip(bars, auc_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement arrow and text
    ax.annotate('', xy=(1, custom_vit_auc), xytext=(0, base_vit_auc),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    
    mid_y = (base_vit_auc + custom_vit_auc) / 2
    ax.text(0.5, mid_y, f'+{improvement_pct:.1f}%\nImprovement',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Add reference line at 0.5 (random classifier)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Classifier (AUC=0.5)')
    
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Fine-tuning on Vision Transformer Performance', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([0.35, 0.65])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'finetuning_impact')
    
    # Print summary
    print(f"\n  Base ViT AUC: {base_vit_auc:.4f}")
    print(f"  Custom ViT AUC: {custom_vit_auc:.4f}")
    print(f"  Improvement: +{improvement_pct:.1f}%")


# =============================================================================
# FIGURE B: Per-Threshold APCER/BPCER Comparison Table
# =============================================================================
def generate_threshold_comparison_table():
    print("\n[Figure B] Generating per-threshold APCER/BPCER comparison table...")
    
    # Load comparison data
    with open(RESULTS_DIR / 'model_comparison.json') as f:
        data = json.load(f)
    
    # Prepare data for table (exclude SigNet_F)
    table_data = []
    
    for item in data['comparison_table']:
        model_name = item['model_name']
        
        # Skip SigNet_F
        if model_name == 'SigNet_F':
            continue
        
        row = {
            'Model': model_name.replace('_', ' '),
            'τ=0.5 APCER': f"{item['t0.5_apcer']*100:.1f}%",
            'τ=0.5 BPCER': f"{item['t0.5_bpcer']*100:.1f}%",
            'τ=0.7 APCER': f"{item['t0.7_apcer']*100:.1f}%",
            'τ=0.7 BPCER': f"{item['t0.7_bpcer']*100:.1f}%",
            'EER': f"{item['eer']*100:.1f}%",
            'EER Threshold': f"{item['eer_threshold']:.3f}"
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.8)
    
    # Style header
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Color-code rows
    colors = ['#E8F4F8', '#FFF3E0', '#F3E5F5']
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i])
            
            # Highlight problematic values
            text = df.iloc[i, j]
            if j in [2, 4]:  # BPCER columns
                try:
                    val = float(str(text).replace('%', ''))
                    if val > 85:  # High BPCER
                        cell.set_facecolor('#FFCDD2')
                        cell.set_text_props(weight='bold', color='red')
                except:
                    pass
            elif j in [1, 3]:  # APCER columns
                try:
                    val = float(str(text).replace('%', ''))
                    if val > 85:  # High APCER
                        cell.set_facecolor('#FFCDD2')
                        cell.set_text_props(weight='bold', color='red')
                except:
                    pass
    
    plt.title('Model Comparison: APCER and BPCER at Multiple Thresholds', 
              fontsize=13, fontweight='bold', pad=20)
    
    # Add footnote
    fig.text(0.5, 0.02, 
             'APCER = Attack Presentation Classification Error Rate (attacks accepted as genuine)\n'
             'BPCER = Bona Fide Presentation Classification Error Rate (genuine rejected as attacks)',
             ha='center', fontsize=8, style='italic', wrap=True)
    
    plt.tight_layout()
    save_figure(fig, 'threshold_comparison_table')
    
    # Also save as CSV
    df.to_csv(FIGURES_DIR / 'threshold_comparison_table.csv', index=False)
    print(f"  Also saved as CSV: threshold_comparison_table.csv")


# =============================================================================
# FIGURE C: Score Distribution Comparison (3 Models - No SigNet_F)
# =============================================================================
def generate_score_distribution_comparison():
    print("\n[Figure C] Generating score distribution comparison...")
    
    # Load distribution data
    with open(RESULTS_DIR / 'score_distribution_analysis.json') as f:
        dist_data = json.load(f)
    
    # Prepare data for violin plots (exclude SigNet_F)
    all_scores = []
    labels = []
    
    model_order = ['Custom_ViT_FineTuned', 'ResNet50_Pretrained', 'Base_ViT_Pretrained']
    model_names = ['Custom ViT\nFine-tuned', 'ResNet-50\nPretrained', 'Base ViT\nPretrained']
    
    for model_key, display_name in zip(model_order, model_names):
        # Load score distributions for each model
        df = pd.read_csv(RESULTS_DIR / model_key / 'score_distributions.csv')
        
        live_scores = df[df['label'] == 'live']['score'].values
        spoof_scores = df[df['label'] == 'spoof']['score'].values
        
        all_scores.extend(live_scores)
        labels.extend([f'{display_name}\n(Live)'] * len(live_scores))
        
        all_scores.extend(spoof_scores)
        labels.extend([f'{display_name}\n(Spoof)'] * len(spoof_scores))
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_plot = pd.DataFrame({'Score': all_scores, 'Model': labels})
    
    # Create position mapping
    unique_labels = []
    for model_name in model_names:
        unique_labels.append(f'{model_name}\n(Live)')
        unique_labels.append(f'{model_name}\n(Spoof)')
    
    positions = list(range(len(unique_labels)))
    
    # Custom violin plot
    parts = ax.violinplot([df_plot[df_plot['Model'] == label]['Score'].values 
                            for label in unique_labels],
                          positions=positions,
                          showmeans=True,
                          showmedians=True,
                          widths=0.7)
    
    # Color the violins
    colors = ['#2E86AB', '#E63946', '#A23B72', '#F77F00', '#95A5A6', '#E74C3C']
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Styling
    parts['cmeans'].set_edgecolor('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_edgecolor('red')
    parts['cmedians'].set_linewidth(2)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
    ax.set_title('Score Distribution Comparison Across Models', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 1.05])
    
    # Add statistics annotations
    for i, model_key in enumerate(model_order):
        model_stats = dist_data[model_key]
        x_pos = i * 2
        
        live_mean = model_stats['live_scores']['mean']
        spoof_mean = model_stats['spoof_scores']['mean']
        
        # Annotate Live
        ax.text(x_pos, live_mean, f'μ={live_mean:.3f}',
                ha='right', va='center', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Annotate Spoof
        ax.text(x_pos + 1, spoof_mean, f'μ={spoof_mean:.3f}',
                ha='left', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Mean (black line)'),
        Patch(facecolor='red', edgecolor='red', label='Median (red line)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'score_distribution_comparison')
    
    # Print statistics
    print("\n  Model Statistics:")
    for model_key in model_order:
        stats = dist_data[model_key]
        print(f"\n  {model_key}:")
        print(f"    Live mean: {stats['live_scores']['mean']:.4f}")
        print(f"    Spoof mean: {stats['spoof_scores']['mean']:.4f}")
        print(f"    Separation (Cohen's d): {stats['separation_metrics']['cohen_d']:.4f}")


# =============================================================================
# FIGURE D: Overlaid Histograms (3 Models - No SigNet_F)
# =============================================================================
def generate_overlaid_histograms():
    print("\n[Figure D] Generating overlaid histogram comparison...")
    
    # Load distribution data
    with open(RESULTS_DIR / 'score_distribution_analysis.json') as f:
        dist_data = json.load(f)
    
    model_order = ['Custom_ViT_FineTuned', 'ResNet50_Pretrained', 'Base_ViT_Pretrained']
    model_names = ['Custom ViT', 'ResNet-50', 'Base ViT']
    colors = ['#2E86AB', '#A23B72', '#F77F00']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (model_key, model_name, color) in enumerate(zip(model_order, model_names, colors)):
        ax = axes[idx]
        
        # Load scores
        df = pd.read_csv(RESULTS_DIR / model_key / 'score_distributions.csv')
        live_scores = df[df['label'] == 'live']['score'].values
        spoof_scores = df[df['label'] == 'spoof']['score'].values
        
        stats = dist_data[model_key]
        
        # Plot histograms
        ax.hist(live_scores, bins=40, alpha=0.6, color='#2E86AB', 
                label=f"Live (μ={stats['live_scores']['mean']:.3f})",
                edgecolor='black', linewidth=0.5)
        ax.hist(spoof_scores, bins=40, alpha=0.6, color='#E63946', 
                label=f"Spoof (μ={stats['spoof_scores']['mean']:.3f})",
                edgecolor='black', linewidth=0.5)
        
        # Add mean lines
        ax.axvline(stats['live_scores']['mean'], color='#2E86AB', 
                   linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(stats['spoof_scores']['mean'], color='#E63946', 
                   linestyle='--', linewidth=2, alpha=0.8)
        
        # Add Cohen's d annotation
        cohen_d = stats['separation_metrics']['cohen_d']
        
        # Determine annotation color and text based on Cohen's d sign
        if cohen_d < 0:
            annotation_color = '#FFCDD2'
            annotation_text = f"Cohen's d = {cohen_d:.3f}\n⚠ Negative = Wrong Direction"
        else:
            annotation_color = 'lightgreen'
            annotation_text = f"Cohen's d = {cohen_d:.3f}"
        
        ax.text(0.95, 0.95, annotation_text,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=annotation_color, alpha=0.8))
        
        ax.set_xlabel('Prediction Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xlim([0, 1])
    
    fig.suptitle('Score Distribution Comparison: Live vs Spoof', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    # Add explanatory note
    fig.text(0.5, 0.01, 
             'Note: Negative Cohen\'s d indicates the model assigns higher scores to Live than Spoof (opposite of desired behavior)',
             ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    save_figure(fig, 'overlaid_histograms_comparison')


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    try:
        generate_finetuning_impact()
        generate_threshold_comparison_table()
        generate_score_distribution_comparison()
        generate_overlaid_histograms()
        
        print("\n" + "="*80)
        print("ALL ADDITIONAL FIGURES GENERATED SUCCESSFULLY")
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
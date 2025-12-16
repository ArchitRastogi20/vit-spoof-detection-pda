# Vision Transformer–Based Face Presentation Attack Detection

This repository contains the code, experiments, and analysis for a Master’s-level research project on **face presentation attack detection (PAD)** using **Vision Transformers (ViT)** with a **differential data augmentation strategy**. The work provides a systematic comparison between a fine-tuned Vision Transformer, a frozen pretrained Vision Transformer, and a CNN baseline (ResNet50) on the CelebA-Spoof dataset.

The goal of this project is not only to report performance metrics, but to **analyze model behavior across operating points**, understand the impact of fine-tuning, and study how score distributions affect deployment decisions in biometric security systems.

---

## Project Overview

Face recognition systems are vulnerable to presentation attacks such as printed photos and replay attacks. This project investigates whether **Vision Transformers**, when properly fine-tuned and trained with imbalance-aware augmentation, can serve as competitive alternatives to CNN-based PAD systems.

Key characteristics of this work:

* Vision Transformer (ViT-B/16) fine-tuned end-to-end for PAD
* Explicit comparison against a pretrained ResNet50 and a frozen ViT baseline
* Differential data augmentation to address severe class imbalance
* Evaluation using **ISO/IEC 30107-compliant metrics**
* Threshold-dependent analysis for real-world deployment scenarios

This repository accompanies the paper:
**“Vision Transformer-Based Presentation Attack Detection with Differential Augmentation.”**

---

## Key Contributions

* **Differential data augmentation**: A class-specific augmentation strategy applying **8× augmentation to live samples** and **2× augmentation to spoof samples**, converting a 1:3.87 imbalance into a near-balanced dataset.
* **Fine-tuning analysis**: Demonstrates that a frozen ImageNet-pretrained ViT performs **worse than random** for PAD, while full fine-tuning yields a **35.5% relative ROC-AUC improvement**.
* **Architecture comparison**: Shows that ResNet50 achieves better Equal Error Rate (EER), while the fine-tuned ViT provides superior ranking ability and broader score distributions.
* **Operating-point evaluation**: Reports APCER and BPCER at multiple thresholds, revealing pathological failures of some models under high-security settings.
* **Deployment realism**: Includes single-image inference benchmarks showing all models exceed **180 FPS**, suitable for real-time PAD systems.

---

## Repository Structure

```
vit-spoof-detection-pda/
├── train_advanced.py              # Main training script (ViT, focal loss, fine-tuning)
├── test.py                        # Evaluation script with PAD metrics
├── augment_data.py                # Differential data augmentation pipeline
├── requirements.txt               # Python dependencies
│
├── data_vis/                      # Figure and analysis scripts
│   ├── generate_figures.py
│   ├── generate_additional_figures.py
│   ├── violin_plot.py
│   ├── make_data_augmentation_figure.py
│   ├── make_data_augmentation_2x2.py
│   └── make_misclassified_2x2.py
│
├── figures/                       # Generated figures used in the paper
│   ├── confusion matrices
│   ├── ROC curves
│   ├── score distributions
│   ├── misclassified samples
│   └── figures_more/              # Extended analysis (threshold tables, comparisons)
│
├── testing_set_analysis_src/      # Post-hoc testing and cross-model evaluation
│   ├── evaluate_all_models.py
│   └── additional_analysis.py
│
├── simple/                        # Baseline implementation
│   ├── train.py                   # Simple training without advanced techniques
│   ├── test.py
│   └── results/
│
├── results/                       # Saved metrics and evaluation outputs
├── docs/                          # Supplementary documentation
├── USAGE.md                       # Usage notes
└── LICENSE
```

---

## Dataset

This project uses the **CelebA-Spoof** dataset.

Due to computational constraints, experiments are conducted on the **first 22 shards out of 100**, which still provide a representative and challenging subset.

### Class Distribution

| Stage              | Live    | Spoof   | Ratio (Live:Spoof) |
| ------------------ | ------- | ------- | ------------------ |
| Original           | ~7,500  | ~29,000 | 1 : 3.87           |
| After Augmentation | ~60,000 | ~58,000 | 1 : 0.97           |

### Splits

* Training: 115,125 images
* Validation: 20,317 images
* Test: 1,747 images (1,076 live / 671 spoof)

The **test set intentionally uses a different class distribution** (live-heavy) to evaluate robustness under deployment-like conditions.

---

## Data Preprocessing and Augmentation

All images are:

* Converted to RGB
* Resized to 224 × 224
* Denoised using fast non-local means
* Normalized with ImageNet statistics

### Differential Augmentation Strategy

To address imbalance without destroying attack artifacts:

* **Live samples**: 8× augmentation
* **Spoof samples**: 2× augmentation

Augmentations include:

* Horizontal flip
* Rotation
* Color jitter
* Gaussian blur and noise
* Perspective distortion
* Elastic deformation
* Sharpness variation

Implementation: `augment_data.py`

---

## Models Evaluated

| Model           | Training Strategy | Parameters | Purpose                   |
| --------------- | ----------------- | ---------- | ------------------------- |
| Custom ViT-B/16 | Fully fine-tuned  | ~86M       | Primary transformer model |
| Base ViT-B/16   | Frozen backbone   | ~86M       | Fine-tuning ablation      |
| ResNet50        | Pretrained CNN    | ~25M       | Strong CNN baseline       |

---

## Training Configuration

* Optimizer: Adam
* Initial learning rate: 1e-5
* Batch size: 64
* Epochs: 120
* Loss: Focal loss with class weighting
* Weight decay: 1e-4

All ViT layers are updated during fine-tuning.

---

## Evaluation Metrics

Evaluation follows **ISO/IEC 30107** standards:

* **APCER** – Attack Presentation Classification Error Rate
* **BPCER** – Bona Fide Presentation Classification Error Rate
* **EER** – Equal Error Rate
* **ROC-AUC** – Ranking performance
* **Inference time** – Single-image latency and FPS

Metrics are reported at:

* Threshold = 0.5 (neutral)
* Threshold = 0.7 (security-focused)
* EER operating point

---

## Key Results Summary

| Metric             | Custom ViT | ResNet50   | Base ViT |
| ------------------ | ---------- | ---------- | -------- |
| ROC-AUC            | 0.5665     | 0.5597     | 0.4181   |
| EER (%)            | 45.26      | **44.05**  | 54.93    |
| Score Distribution | Broad      | Compressed | Inverted |

**Important finding:** the frozen ViT learns an inverted decision function and performs worse than random.

---

## Inference Speed

All models achieve real-time performance on an NVIDIA RTX A4500:

* Custom ViT: ~184 FPS
* ResNet50: ~202 FPS
* Base ViT: ~206 FPS

---

## Practical Recommendations

* Use **ResNet50** for balanced, efficiency-focused deployments.
* Use **Custom ViT** for high-security scenarios requiring flexible threshold selection.
* **Never deploy frozen pretrained transformers** for PAD tasks.

---

## Limitations

* Only 22% of CelebA-Spoof was used
* No cross-dataset generalization study
* Absolute performance leaves room for improvement

---



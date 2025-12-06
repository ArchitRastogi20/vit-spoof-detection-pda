# Vision Transformer for Face Anti-Spoofing

A fine-tuned Vision Transformer (ViT-B/16) implementation for presentation attack detection in face recognition systems, trained on the CelebA-Spoof dataset.

## Overview

This repository contains a comprehensive face anti-spoofing solution using Vision Transformers with advanced training techniques including Focal Loss, class weighting, and GPU-accelerated data augmentation. The model achieves strong performance with 95.61% AUC-ROC and 10.83% EER.

## Key Features

- Fine-tuned ViT-B/16 architecture pre-trained on ImageNet-21k
- GPU-accelerated data augmentation pipeline using Kornia
- Focal Loss with class weighting for imbalanced data handling
- Comprehensive evaluation with biometric-specific metrics (APCER, BPCER, EER)
- Threshold optimization for operational deployment
- Complete training and testing pipeline with visualization tools

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 83.29% |
| AUC-ROC | 95.61% |
| F1-Score | 87.80% |
| EER | 10.83% |
| APCER (at 0.5) | 39.79% |
| BPCER (at 0.5) | 2.32% |

## Repository Structure

```
vit-spoof-detection-pda/
├── train_advanced.py          # Main training script with Focal Loss
├── test.py                    # Testing and evaluation script
├── augment_data.py            # GPU-accelerated data augmentation
├── data_vis.py                # APCER/BPCER curve visualization
├── results/                   # Evaluation results and metrics
│   ├── apcer_bpcer_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── test_metrics.csv
├── simple/                    # Baseline implementation
└── docs/                      # Documentation and reports
```

## Installation

```bash
pip install -U torch torchvision transformers timm kornia scikit-learn pandas numpy seaborn matplotlib pillow
```

## Usage

### Data Preparation

```bash
python augment_data.py
```

Configure paths in the script:
- `input_dir`: Path to CelebA-Spoof dataset
- `output_dir`: Path for augmented images
- `live_augmentations`: Augmentation multiplier for live samples
- `spoof_augmentations`: Augmentation multiplier for spoof samples

### Training

```bash
python train_advanced.py
```

Key hyperparameters:
- Model: ViT-B/16 (86M parameters)
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.05)
- Batch size: 128
- Epochs: 50 with early stopping

### Testing

```bash
python test.py
```

Configure checkpoint path in script:
- `checkpoint_path`: Path to trained model checkpoint

### Visualization

Generate APCER vs BPCER trade-off curve:

```bash
python data_vis.py
```

## Model Architecture

The model consists of:
1. ViT-B/16 backbone (768-dimensional embeddings)
2. Layer normalization
3. Dropout (p=0.1)
4. Fully connected layer (768 → 512) with GELU activation
5. Dropout (p=0.1)
6. Output layer (512 → 2)

Total parameters: 86M (all trainable)

## Dataset

Trained on CelebA-Spoof dataset subset (first 22 of 100 shards):
- Original: 7,500 live images, 29,000 spoof images
- After augmentation: 60,497 live, 74,945 spoof images
- Train: 115,125 images (85%)
- Validation: 20,317 images (15%)
- Test: 1,747 images (held-out subjects)

## Pre-trained Model

The trained model is available on Hugging Face:
[ArchitRastogi/vit-spoof-detection-pda](https://huggingface.co/ArchitRastogi/vit-spoof-detection-pda)

## Evaluation Metrics

The system is evaluated using ISO/IEC 30107-3 standard metrics:

- **APCER** (Attack Presentation Classification Error Rate): Proportion of attacks incorrectly accepted
- **BPCER** (Bona Fide Presentation Classification Error Rate): Proportion of genuine presentations incorrectly rejected
- **EER** (Equal Error Rate): Operating point where APCER equals BPCER
- **AUC-ROC**: Area under receiver operating characteristic curve

## Results

Detailed results are available in the `results/` directory:
- Confusion matrices at different thresholds
- ROC curves
- APCER vs BPCER trade-off analysis
- Per-subject performance breakdown
- Per-image predictions with confidence scores

## Baseline Comparison

The `simple/` directory contains a baseline implementation using standard cross-entropy loss without advanced techniques, demonstrating the effectiveness of the improvements in the main approach.

## License

Apache License 2.0

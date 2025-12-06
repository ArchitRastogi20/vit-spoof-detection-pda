# Baseline Implementation

This directory contains a baseline implementation of face anti-spoofing using Vision Transformers with standard training techniques, serving as a comparison point for the advanced implementation in the main directory.

## Approach

This baseline uses:
- ViT-B/16 architecture pre-trained on ImageNet-21k
- Standard cross-entropy loss
- Basic data augmentation (random crops, flips, color jitter)
- No class weighting or focal loss
- AdamW optimizer with cosine annealing

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 76.94% |
| AUC-ROC | 96.13% |
| F1-Score | 59.02% |
| EER | 10.34% |
| APCER (at 0.5) | 27.00% |
| BPCER (at 0.5) | 4.32% |

## Dataset

- Test: 2,402 images (417 live, 1,985 spoof)
- Different class distribution compared to advanced implementation

## Comparison with Advanced Implementation

The advanced implementation in the main directory achieves superior performance through:

1. **Focal Loss**: Better handling of class imbalance and hard examples
2. **GPU-accelerated augmentation**: More diverse and extensive augmentation using Kornia
3. **Class weighting**: Explicit handling of imbalanced training data
4. **Threshold optimization**: Systematic analysis of operating points

## Key Differences

| Aspect | Baseline | Advanced |
|--------|----------|----------|
| Loss Function | Cross-Entropy | Focal Loss |
| Augmentation | CPU (torchvision) | GPU (Kornia) |
| Class Handling | None | Weighted + Focal |
| Accuracy | 76.94% | 83.29% |
| F1-Score | 59.02% | 87.80% |

## Usage

### Training

```bash
python train.py
```

### Testing

```bash
python test.py
```

Configure checkpoint path in the script before running.

## Results

All evaluation results are stored in the `results/` subdirectory:
- `confusion_matrix.png`: Visual confusion matrix
- `roc_curve.png`: ROC curve with AUC
- `test_metrics.csv`: Comprehensive metrics
- `test_summary.txt`: Text summary of results
- `per_subject_results.csv`: Per-subject accuracy breakdown

## Conclusion

This baseline demonstrates that while standard ViT fine-tuning achieves reasonable performance (96.13% AUC-ROC), the advanced techniques in the main implementation provide substantial improvements in practical metrics, particularly accuracy (76.94% → 83.29%) and F1-score (59.02% → 87.80%).

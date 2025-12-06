import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
import timm
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    classification_report,
    roc_curve
)
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================
class TestConfig:
    # Paths
    test_data_root = "./test_split"
    checkpoint_path = "checkpoints_advanced/best_model_run_eif1jakb.pth"
    output_dir = "./test_results"
    
    # Model
    model_name = "vit_base_patch16_224"
    num_classes = 2
    dropout = 0.1
    
    # Testing
    batch_size = 128
    num_workers = 28
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4
    mixed_precision = True
    
    # Image
    img_size = 224
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== MODEL DEFINITION ====================
class ViTFaceAntiSpoofing(nn.Module):
    """Same architecture as training"""
    def __init__(self, config):
        super().__init__()
        self.vit = timm.create_model(config.model_name, pretrained=False, num_classes=0)
        embed_dim = self.vit.num_features if hasattr(self.vit, 'num_features') else 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


# ==================== DATASET ====================
class TestDataset(Dataset):
    """Dataset for test split"""
    def __init__(self, test_root, transform=None):
        self.test_root = Path(test_root)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.subjects = []
        self.image_names = []
        
        logger.info(f"Scanning test dataset at {test_root}")
        
        # Scan all subject directories
        for subject_dir in sorted(self.test_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            
            # Process live images
            live_dir = subject_dir / "live"
            if live_dir.exists():
                for img_path in sorted(live_dir.glob("*.png")):
                    self.samples.append(str(img_path))
                    self.labels.append(1)  # 1 = Live
                    self.subjects.append(subject_id)
                    self.image_names.append(img_path.name)
            
            # Process spoof images
            spoof_dir = subject_dir / "spoof"
            if spoof_dir.exists():
                for img_path in sorted(spoof_dir.glob("*.png")):
                    self.samples.append(str(img_path))
                    self.labels.append(0)  # 0 = Spoof
                    self.subjects.append(subject_id)
                    self.image_names.append(img_path.name)
        
        live_count = sum(self.labels)
        spoof_count = len(self.labels) - live_count
        
        logger.info(f"Loaded {len(self.samples)} test images")
        logger.info(f"  - Live: {live_count} ({live_count/len(self.labels)*100:.2f}%)")
        logger.info(f"  - Spoof: {spoof_count} ({spoof_count/len(self.labels)*100:.2f}%)")
        logger.info(f"  - Subjects: {len(set(self.subjects))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx


def get_test_transforms(img_size=224):
    """Test time transforms - no augmentation"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ==================== TESTING ====================
def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Log checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    
    logger.info(f"Checkpoint loaded successfully")
    logger.info(f"  - Trained epoch: {epoch}")
    if metrics:
        logger.info(f"  - Training metrics: {metrics}")
    
    return model, checkpoint


@torch.no_grad()
def test_model(model, loader, config):
    """Run inference on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []
    
    logger.info("Running inference on test set...")
    
    pbar = tqdm(loader, desc="Testing")
    
    for images, labels, indices in pbar:
        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        with autocast(enabled=config.mixed_precision):
            outputs = model(images)
        
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being live
        all_indices.extend(indices.numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_indices)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    logger.info("\n" + "="*60)
    logger.info("CALCULATING METRICS")
    logger.info("="*60)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
    
    # False acceptance rate and false rejection rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Equal error rate (approximate)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'specificity': specificity,
        'npv': npv,
        'ppv': ppv,
        'far': far,
        'frr': frr,
        'eer': eer,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_samples': len(y_true),
        'live_samples': int(np.sum(y_true)),
        'spoof_samples': int(len(y_true) - np.sum(y_true))
    }
    
    return metrics, cm


def print_metrics(metrics):
    """Pretty print metrics to stdout"""
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS")
    logger.info("="*60)
    
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  AUC-ROC:         {metrics['auc']:.4f}")
    logger.info(f"  F1-Score:        {metrics['f1_score']:.4f}")
    
    logger.info(f"\nDetection Metrics (Live Class):")
    logger.info(f"  Precision (PPV): {metrics['precision']:.4f}")
    logger.info(f"  Recall (TPR):    {metrics['recall']:.4f}")
    logger.info(f"  Specificity:     {metrics['specificity']:.4f}")
    logger.info(f"  NPV:             {metrics['npv']:.4f}")
    
    logger.info(f"\nError Rates:")
    logger.info(f"  FAR (FPR):       {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
    logger.info(f"  FRR (FNR):       {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
    logger.info(f"  EER:             {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {metrics['tn']}")
    logger.info(f"  False Positives: {metrics['fp']}")
    logger.info(f"  False Negatives: {metrics['fn']}")
    logger.info(f"  True Positives:  {metrics['tp']}")
    
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total Samples:   {metrics['total_samples']}")
    logger.info(f"  Live Samples:    {metrics['live_samples']}")
    logger.info(f"  Spoof Samples:   {metrics['spoof_samples']}")
    
    logger.info("\n" + "="*60)


def save_results(metrics, cm, y_true, y_pred, y_prob, dataset, indices, output_dir):
    """Save results to CSV and generate plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save overall metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / f"test_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nMetrics saved to: {metrics_path}")
    
    # Save per-image results
    results_data = []
    for idx in indices:
        results_data.append({
            'image_path': dataset.samples[idx],
            'image_name': dataset.image_names[idx],
            'subject_id': dataset.subjects[idx],
            'true_label': 'live' if dataset.labels[idx] == 1 else 'spoof',
            'predicted_label': 'live' if y_pred[np.where(indices == idx)[0][0]] == 1 else 'spoof',
            'probability_live': y_prob[np.where(indices == idx)[0][0]],
            'probability_spoof': 1 - y_prob[np.where(indices == idx)[0][0]],
            'correct': y_true[np.where(indices == idx)[0][0]] == y_pred[np.where(indices == idx)[0][0]]
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = output_dir / f"per_image_results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-image results saved to: {results_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        columns=['Predicted Spoof', 'Predicted Live'],
        index=['Actual Spoof', 'Actual Live']
    )
    cm_path = output_dir / f"confusion_matrix_{timestamp}.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Generate and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Spoof', 'Live'],
                yticklabels=['Spoof', 'Live'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plot_path = output_dir / f"confusion_matrix_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {plot_path}")
    
    # Generate ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = output_dir / f"roc_curve_{timestamp}.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to: {roc_path}")
    except Exception as e:
        logger.warning(f"Could not generate ROC curve: {e}")
    
    # Per-subject analysis
    subject_results = results_df.groupby('subject_id').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    subject_results.columns = ['correct_predictions', 'total_images', 'accuracy']
    subject_results = subject_results.sort_values('accuracy')
    subject_path = output_dir / f"per_subject_results_{timestamp}.csv"
    subject_results.to_csv(subject_path)
    logger.info(f"Per-subject results saved to: {subject_path}")
    
    # Summary report
    summary_path = output_dir / f"test_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FACE ANTI-SPOOFING TEST REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: checkpoints/best_model_run_8f3ihjcw.pth\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"AUC-ROC:         {metrics['auc']:.4f}\n")
        f.write(f"F1-Score:        {metrics['f1_score']:.4f}\n\n")
        
        f.write("DETECTION METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision (PPV): {metrics['precision']:.4f}\n")
        f.write(f"Recall (TPR):    {metrics['recall']:.4f}\n")
        f.write(f"Specificity:     {metrics['specificity']:.4f}\n")
        f.write(f"NPV:             {metrics['npv']:.4f}\n\n")
        
        f.write("ERROR RATES\n")
        f.write("-"*60 + "\n")
        f.write(f"FAR (FPR):       {metrics['far']:.4f} ({metrics['far']*100:.2f}%)\n")
        f.write(f"FRR (FNR):       {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)\n")
        f.write(f"EER:             {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*60 + "\n")
        f.write(f"True Negatives:  {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"True Positives:  {metrics['tp']}\n\n")
        
        f.write("DATASET INFO\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Samples:   {metrics['total_samples']}\n")
        f.write(f"Live Samples:    {metrics['live_samples']}\n")
        f.write(f"Spoof Samples:   {metrics['spoof_samples']}\n")
    
    logger.info(f"Summary report saved to: {summary_path}")
    
    return metrics_path, results_path


# ==================== MAIN ====================
def main():
    config = TestConfig()
    
    logger.info("\n" + "="*60)
    logger.info("FACE ANTI-SPOOFING MODEL TESTING")
    logger.info("="*60)
    logger.info(f"Checkpoint: {config.checkpoint_path}")
    logger.info(f"Test data: {config.test_data_root}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    
    # Check if checkpoint exists
    if not os.path.exists(config.checkpoint_path):
        logger.error(f"Checkpoint not found at {config.checkpoint_path}")
        return
    
    # Create test dataset
    test_transform = get_test_transforms(config.img_size)
    test_dataset = TestDataset(config.test_data_root, transform=test_transform)
    
    if len(test_dataset) == 0:
        logger.error("No test images found!")
        return
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    # Create model
    logger.info("\nInitializing model...")
    model = ViTFaceAntiSpoofing(config).to(config.device)
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(config.checkpoint_path, model, config.device)
    
    # Run testing
    y_pred, y_true, y_prob, indices = test_model(model, test_loader, config)
    
    # Calculate metrics
    metrics, cm = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print metrics to stdout
    print_metrics(metrics)
    
    # Save results
    logger.info("\nSaving results...")
    metrics_path, results_path = save_results(
        metrics, cm, y_true, y_pred, y_prob, 
        test_dataset, indices, config.output_dir
    )
    
    logger.info("\n" + "="*60)
    logger.info("TESTING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"\nAll results saved to: {config.output_dir}/")
    logger.info(f"  - Metrics CSV: {metrics_path}")
    logger.info(f"  - Per-image results: {results_path}")


if __name__ == "__main__":
    main()
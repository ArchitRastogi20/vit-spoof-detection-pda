import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None, processor=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.processor = processor
        self.samples = []
        
        live_dir = self.root_dir / 'live'
        for img_path in live_dir.glob('*.png'):
            self.samples.append((str(img_path), 0))
        
        spoof_dir = self.root_dir / 'spoof'
        for img_path in spoof_dir.glob('*.png'):
            self.samples.append((str(img_path), 1))
        
        logger.info(f"Loaded {len(self.samples)} samples (Live: {len(list(live_dir.glob('*.png')))}, Spoof: {len(list(spoof_dir.glob('*.png')))})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'label': label,
                'path': img_path
            }
        elif self.transform is not None:
            image = self.transform(image)
            return {
                'image': image,
                'label': label,
                'path': img_path
            }
        else:
            raise ValueError("Either processor or transform must be provided")


class ResNet50Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
    
    def forward(self, x):
        return self.resnet(x)


def calculate_metrics_at_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'apcer': apcer,
        'bpcer': bpcer,
        'far': far,
        'frr': frr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_eer_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def evaluate_model_generic(model, dataloader, device, model_name, is_vit=False):
    model.eval()
    
    all_labels = []
    all_scores = []
    all_paths = []
    
    logger.info(f"Starting inference for {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference [{model_name}]"):
            if is_vit:
                pixel_values = batch['pixel_values'].to(device)
                outputs = model(pixel_values)
                logits = outputs.logits
            else:
                images = batch['image'].to(device)
                logits = model(images)
            
            labels = batch['label'].numpy()
            paths = batch['path']
            
            probs = torch.softmax(logits, dim=-1)
            spoof_scores = probs[:, 1].cpu().numpy()
            
            all_labels.extend(labels)
            all_scores.extend(spoof_scores)
            all_paths.extend(paths)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    logger.info(f"Inference completed on {len(all_labels)} samples")
    
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = find_eer_threshold(all_labels, all_scores)
    
    logger.info(f"{model_name} - ROC AUC: {roc_auc:.4f}, EER: {eer:.4f} at threshold {eer_threshold:.4f}")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, float(eer_threshold)]
    results = [calculate_metrics_at_threshold(all_labels, all_scores, t) for t in thresholds]
    
    return {
        'labels': all_labels,
        'scores': all_scores,
        'paths': all_paths,
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'threshold_results': results
    }


def save_model_results(evaluation_results, model_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_labels = evaluation_results['labels']
    all_scores = evaluation_results['scores']
    all_paths = evaluation_results['paths']
    
    per_image_df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': all_labels,
        'spoof_score': all_scores,
        'predicted_label_0.5': (all_scores >= 0.5).astype(int)
    })
    per_image_df.to_csv(output_dir / 'per_image_predictions.csv', index=False)
    
    roc_df = pd.DataFrame({
        'fpr': evaluation_results['fpr'],
        'tpr': evaluation_results['tpr'],
        'threshold': evaluation_results['roc_thresholds']
    })
    roc_df.to_csv(output_dir / 'roc_curve_data.csv', index=False)
    
    results_df = pd.DataFrame(evaluation_results['threshold_results'])
    results_df = results_df.sort_values('threshold')
    results_df.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    
    key_thresholds = [0.5, 0.7, float(evaluation_results['eer_threshold'])]
    confusion_matrices = {}
    
    for threshold in key_thresholds:
        y_pred = (all_scores >= threshold).astype(int)
        cm = confusion_matrix(all_labels, y_pred, labels=[0, 1])
        confusion_matrices[f'threshold_{threshold:.3f}'] = {
            'confusion_matrix': cm.tolist(),
            'threshold': threshold
        }
    
    with open(output_dir / 'confusion_matrices.json', 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    
    summary = {
        'model_name': model_name,
        'evaluation_timestamp': datetime.now().isoformat(),
        'total_samples': len(all_labels),
        'live_samples': int((all_labels == 0).sum()),
        'spoof_samples': int((all_labels == 1).sum()),
        'roc_auc': float(evaluation_results['roc_auc']),
        'eer': float(evaluation_results['eer']),
        'eer_threshold': float(evaluation_results['eer_threshold']),
        'score_statistics': {
            'mean': float(all_scores.mean()),
            'std': float(all_scores.std()),
            'min': float(all_scores.min()),
            'max': float(all_scores.max())
        },
        'operating_points': {
            'threshold_0.5': evaluation_results['threshold_results'][4],
            'threshold_0.7': evaluation_results['threshold_results'][6],
            'eer_point': evaluation_results['threshold_results'][-1]
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("FACE ANTI-SPOOFING EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("DATASET INFO\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Samples:   {len(all_labels)}\n")
        f.write(f"Live Samples:    {(all_labels == 0).sum()}\n")
        f.write(f"Spoof Samples:   {(all_labels == 1).sum()}\n\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"ROC AUC:         {evaluation_results['roc_auc']:.4f}\n")
        f.write(f"EER:             {evaluation_results['eer']:.4f} ({evaluation_results['eer']*100:.2f}%)\n")
        f.write(f"EER Threshold:   {evaluation_results['eer_threshold']:.4f}\n\n")
        f.write("METRICS AT KEY THRESHOLDS\n")
        f.write("-"*60 + "\n\n")
        
        for threshold in [0.5, 0.7, float(evaluation_results['eer_threshold'])]:
            thresholds = [r['threshold'] for r in evaluation_results['threshold_results']]
            idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i]-threshold))
            metrics = evaluation_results['threshold_results'][idx]
            
            f.write(f"Threshold: {metrics['threshold']:.4f}\n")
            f.write(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  F1-Score:        {metrics['f1_score']:.4f}\n")
            f.write(f"  Precision:       {metrics['precision']:.4f}\n")
            f.write(f"  Recall:          {metrics['recall']:.4f}\n")
            f.write(f"  APCER (FAR):     {metrics['apcer']:.4f} ({metrics['apcer']*100:.2f}%)\n")
            f.write(f"  BPCER (FRR):     {metrics['bpcer']:.4f} ({metrics['bpcer']*100:.2f}%)\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TN: {metrics['tn']}, FP: {metrics['fp']}\n")
            f.write(f"    FN: {metrics['fn']}, TP: {metrics['tp']}\n\n")
    
    logger.info(f"Saved detailed results for {model_name} to {output_dir}")


def load_custom_vit_model(device):
    logger.info("Loading custom ViT model (ArchitRastogi/vit-spoof-detection-pda)")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    repo_id = "ArchitRastogi/vit-spoof-detection-pda"
    weights_path = hf_hub_download(repo_id=repo_id, filename="best_model_run_eif1jakb.pth")
    state_dict = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, processor, True


def load_base_vit_model(device):
    logger.info("Loading base ViT model (google/vit-base-patch16-224)")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()
    
    return model, processor, True


def load_resnet50_model(device):
    logger.info("Loading ResNet50 model (pretrained on ImageNet)")
    model = ResNet50Classifier(pretrained=True)
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform, False


def load_signet_model(device):
    logger.info("Attempting to load SigNet-F model")
    try:
        logger.warning("SigNet-F is not publicly available. Using placeholder ResNet50 architecture.")
        logger.warning("For actual SigNet-F results, please provide the model weights.")
        
        model = ResNet50Classifier(pretrained=False)
        model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return model, transform, False
    except Exception as e:
        logger.error(f"Failed to load SigNet-F: {e}")
        return None, None, None


def create_comparison_reports(all_results, output_dir):
    comparison_data = []
    
    for model_name, results in all_results.items():
        eer_metrics = results['threshold_results'][-1]
        t05_metrics = results['threshold_results'][4]
        t07_metrics = results['threshold_results'][6]
        
        comparison_data.append({
            'model_name': model_name,
            'roc_auc': results['roc_auc'],
            'eer': results['eer'],
            'eer_threshold': results['eer_threshold'],
            'eer_accuracy': eer_metrics['accuracy'],
            'eer_apcer': eer_metrics['apcer'],
            'eer_bpcer': eer_metrics['bpcer'],
            'eer_f1': eer_metrics['f1_score'],
            't0.5_accuracy': t05_metrics['accuracy'],
            't0.5_apcer': t05_metrics['apcer'],
            't0.5_bpcer': t05_metrics['bpcer'],
            't0.5_f1': t05_metrics['f1_score'],
            't0.7_accuracy': t07_metrics['accuracy'],
            't0.7_apcer': t07_metrics['apcer'],
            't0.7_bpcer': t07_metrics['bpcer'],
            't0.7_f1': t07_metrics['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    logger.info("\nMODEL COMPARISON SUMMARY")
    logger.info("="*80)
    for _, row in comparison_df.iterrows():
        logger.info(f"{row['model_name']:40s} | AUC: {row['roc_auc']:.4f} | EER: {row['eer']:.4f} | Acc@0.5: {row['t0.5_accuracy']:.4f}")
    logger.info("="*80)
    
    comparison_json = {
        'comparison_timestamp': datetime.now().isoformat(),
        'models_evaluated': list(all_results.keys()),
        'comparison_table': comparison_data,
        'best_model_by_auc': comparison_df.iloc[0]['model_name'],
        'best_model_by_eer': comparison_df.loc[comparison_df['eer'].idxmin()]['model_name'],
        'best_model_by_accuracy': comparison_df.loc[comparison_df['t0.5_accuracy'].idxmax()]['model_name']
    }
    
    with open(output_dir / 'model_comparison.json', 'w') as f:
        json.dump(comparison_json, f, indent=2)
    
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Dataset: test_inf/\n")
        f.write(f"Total Samples: {len(results['labels'])}\n\n")
        
        f.write("RANKING BY ROC AUC\n")
        f.write("-"*80 + "\n")
        for idx, (_, row) in enumerate(comparison_df.iterrows(), 1):
            f.write(f"{idx}. {row['model_name']:40s} AUC: {row['roc_auc']:.4f}\n")
        
        f.write("\n\nDETAILED COMPARISON AT THRESHOLD = 0.5\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<40s} {'Accuracy':<12s} {'APCER':<12s} {'BPCER':<12s} {'F1':<12s}\n")
        f.write("-"*80 + "\n")
        for _, row in comparison_df.iterrows():
            f.write(f"{row['model_name']:<40s} {row['t0.5_accuracy']:<12.4f} {row['t0.5_apcer']:<12.4f} {row['t0.5_bpcer']:<12.4f} {row['t0.5_f1']:<12.4f}\n")
        
        f.write("\n\nDETAILED COMPARISON AT EER POINT\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<40s} {'EER':<12s} {'Threshold':<12s} {'Accuracy':<12s} {'F1':<12s}\n")
        f.write("-"*80 + "\n")
        for _, row in comparison_df.iterrows():
            f.write(f"{row['model_name']:<40s} {row['eer']:<12.4f} {row['eer_threshold']:<12.4f} {row['eer_accuracy']:<12.4f} {row['eer_f1']:<12.4f}\n")
    
    logger.info(f"Comparison reports saved to {output_dir}")


def main():
    output_dir = Path('results_inf_final')
    output_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / 'evaluation_pipeline.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info("MULTI-MODEL EVALUATION PIPELINE")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    models_to_evaluate = [
        ('Custom_ViT_FineTuned', load_custom_vit_model),
        ('Base_ViT_Pretrained', load_base_vit_model),
        ('ResNet50_Pretrained', load_resnet50_model),
        ('SigNet_F', load_signet_model)
    ]
    
    all_results = {}
    
    for model_name, loader_func in models_to_evaluate:
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING: {model_name}")
        logger.info(f"{'='*80}")
        
        try:
            model, processor_or_transform, is_vit = loader_func(device)
            
            if model is None:
                logger.warning(f"Skipping {model_name} - model not available")
                continue
            
            if is_vit:
                dataset = SpoofDataset('test_inf', processor=processor_or_transform)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            else:
                dataset = SpoofDataset('test_inf', transform=processor_or_transform)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            
            evaluation_results = evaluate_model_generic(
                model, dataloader, device, model_name, is_vit=is_vit
            )
            
            model_output_dir = output_dir / model_name
            save_model_results(evaluation_results, model_name, model_output_dir)
            
            all_results[model_name] = evaluation_results
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("CREATING COMPARISON REPORTS")
        logger.info(f"{'='*80}")
        create_comparison_reports(all_results, output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION PIPELINE COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info(f"\nEvaluated {len(all_results)} models successfully")


if __name__ == "__main__":
    main()

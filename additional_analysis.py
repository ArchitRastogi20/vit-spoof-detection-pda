import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from PIL import Image
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_score_distributions(results_dir):
    """Analyze score distributions for all models"""
    logger.info("Analyzing score distributions...")
    
    results_dir = Path(results_dir)
    all_distributions = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        pred_file = model_dir / 'per_image_predictions.csv'
        if not pred_file.exists():
            continue
        
        df = pd.read_csv(pred_file)
        
        live_scores = df[df['true_label'] == 0]['spoof_score'].values
        spoof_scores = df[df['true_label'] == 1]['spoof_score'].values
        
        live_above_median_spoof = np.sum(live_scores > np.median(spoof_scores))
        spoof_below_median_live = np.sum(spoof_scores < np.median(live_scores))
        total_overlap = live_above_median_spoof + spoof_below_median_live
        total_samples = len(live_scores) + len(spoof_scores)
        
        distribution_stats = {
            'model_name': model_dir.name,
            'live_scores': {
                'mean': float(np.mean(live_scores)),
                'std': float(np.std(live_scores)),
                'median': float(np.median(live_scores)),
                'min': float(np.min(live_scores)),
                'max': float(np.max(live_scores)),
                'q25': float(np.percentile(live_scores, 25)),
                'q75': float(np.percentile(live_scores, 75))
            },
            'spoof_scores': {
                'mean': float(np.mean(spoof_scores)),
                'std': float(np.std(spoof_scores)),
                'median': float(np.median(spoof_scores)),
                'min': float(np.min(spoof_scores)),
                'max': float(np.max(spoof_scores)),
                'q25': float(np.percentile(spoof_scores, 25)),
                'q75': float(np.percentile(spoof_scores, 75))
            },
            'separation_metrics': {
                'mean_difference': float(np.mean(spoof_scores) - np.mean(live_scores)),
                'cohen_d': float((np.mean(spoof_scores) - np.mean(live_scores)) / 
                               np.sqrt((np.std(spoof_scores)**2 + np.std(live_scores)**2) / 2)),
                'overlap_percentage': float(total_overlap / total_samples * 100)
            }
        }
        
        all_distributions[model_dir.name] = distribution_stats
        
        dist_df = pd.DataFrame({
            'score': np.concatenate([live_scores, spoof_scores]),
            'label': ['live'] * len(live_scores) + ['spoof'] * len(spoof_scores),
            'label_numeric': [0] * len(live_scores) + [1] * len(spoof_scores)
        })
        dist_df.to_csv(model_dir / 'score_distributions.csv', index=False)
        
        logger.info(f"{model_dir.name}: Live mean={distribution_stats['live_scores']['mean']:.4f}, "
                   f"Spoof mean={distribution_stats['spoof_scores']['mean']:.4f}, "
                   f"Separation={distribution_stats['separation_metrics']['mean_difference']:.4f}")
    
    with open(results_dir / 'score_distribution_analysis.json', 'w') as f:
        json.dump(all_distributions, f, indent=2)
    
    comparison_df = pd.DataFrame([
        {
            'model': d['model_name'],
            'live_mean': d['live_scores']['mean'],
            'spoof_mean': d['spoof_scores']['mean'],
            'mean_diff': d['separation_metrics']['mean_difference'],
            'cohen_d': d['separation_metrics']['cohen_d'],
            'overlap_pct': d['separation_metrics']['overlap_percentage']
        }
        for d in all_distributions.values()
    ])
    comparison_df = comparison_df.sort_values('cohen_d', ascending=False)
    comparison_df.to_csv(results_dir / 'score_separation_comparison.csv', index=False)
    
    logger.info("Score distribution analysis completed")
    return all_distributions


def extract_failed_cases(results_dir, test_data_dir, top_n=20):
    """Extract worst misclassified examples for each model"""
    logger.info(f"Extracting top {top_n} failed cases for each model...")
    
    results_dir = Path(results_dir)
    test_data_dir = Path(test_data_dir)
    failed_cases_dir = results_dir / 'failed_cases_analysis'
    failed_cases_dir.mkdir(exist_ok=True)
    
    all_failed_cases = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == 'failed_cases_analysis':
            continue
        
        pred_file = model_dir / 'per_image_predictions.csv'
        if not pred_file.exists():
            continue
        
        df = pd.read_csv(pred_file)
        
        df['error'] = np.abs(df['true_label'] - df['spoof_score'])
        df['prediction_0.5'] = (df['spoof_score'] >= 0.5).astype(int)
        df['is_correct'] = (df['prediction_0.5'] == df['true_label'])
        
        false_positives = df[(df['true_label'] == 0) & (df['prediction_0.5'] == 1)].nlargest(top_n, 'spoof_score')
        false_negatives = df[(df['true_label'] == 1) & (df['prediction_0.5'] == 0)].nsmallest(top_n, 'spoof_score')
        
        model_failed_dir = failed_cases_dir / model_dir.name
        model_failed_dir.mkdir(exist_ok=True)
        
        fp_dir = model_failed_dir / 'false_positives'
        fn_dir = model_failed_dir / 'false_negatives'
        fp_dir.mkdir(exist_ok=True)
        fn_dir.mkdir(exist_ok=True)
        
        fp_cases = []
        for idx, row in false_positives.iterrows():
            src_path = Path(row['image_path'])
            if src_path.exists():
                dst_path = fp_dir / f"FP_score{row['spoof_score']:.4f}_{src_path.name}"
                shutil.copy2(src_path, dst_path)
                fp_cases.append({
                    'image': src_path.name,
                    'true_label': 'live',
                    'predicted_score': row['spoof_score'],
                    'error_magnitude': row['error']
                })
        
        fn_cases = []
        for idx, row in false_negatives.iterrows():
            src_path = Path(row['image_path'])
            if src_path.exists():
                dst_path = fn_dir / f"FN_score{row['spoof_score']:.4f}_{src_path.name}"
                shutil.copy2(src_path, dst_path)
                fn_cases.append({
                    'image': src_path.name,
                    'true_label': 'spoof',
                    'predicted_score': row['spoof_score'],
                    'error_magnitude': row['error']
                })
        
        all_failed_cases[model_dir.name] = {
            'false_positives': fp_cases,
            'false_negatives': fn_cases,
            'total_fp': len(false_positives),
            'total_fn': len(false_negatives)
        }
        
        pd.DataFrame(fp_cases).to_csv(model_failed_dir / 'false_positives.csv', index=False)
        pd.DataFrame(fn_cases).to_csv(model_failed_dir / 'false_negatives.csv', index=False)
        
        logger.info(f"{model_dir.name}: Extracted {len(fp_cases)} FP and {len(fn_cases)} FN cases")
    
    with open(failed_cases_dir / 'failed_cases_summary.json', 'w') as f:
        json.dump(all_failed_cases, f, indent=2)
    
    logger.info(f"Failed cases extracted to {failed_cases_dir}")
    return all_failed_cases


def benchmark_inference_time(results_dir, test_data_dir, num_samples=100):
    """Benchmark inference time for each model"""
    logger.info(f"Benchmarking inference time on {num_samples} samples...")
    
    from transformers import ViTForImageClassification, ViTImageProcessor
    from torchvision import models, transforms
    from huggingface_hub import hf_hub_download
    import torch.nn as nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(results_dir)
    test_data_dir = Path(test_data_dir)
    
    live_images = list((test_data_dir / 'live').glob('*.png'))
    spoof_images = list((test_data_dir / 'spoof').glob('*.png'))
    
    sample_images = live_images[:num_samples//2] + spoof_images[:num_samples//2]
    
    benchmark_results = {}
    
    class ResNet50Classifier(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.resnet = models.resnet50(pretrained=pretrained)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        def forward(self, x):
            return self.resnet(x)
    
    models_config = {
        'Custom_ViT_FineTuned': {
            'loader': lambda: (
                ViTImageProcessor.from_pretrained("google/vit-base-patch16-224"),
                ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True),
                True
            ),
            'weights': 'ArchitRastogi/vit-spoof-detection-pda'
        },
        'Base_ViT_Pretrained': {
            'loader': lambda: (
                ViTImageProcessor.from_pretrained("google/vit-base-patch16-224"),
                ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True),
                True
            ),
            'weights': None
        },
        'ResNet50_Pretrained': {
            'loader': lambda: (
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                ResNet50Classifier(pretrained=True),
                False
            ),
            'weights': None
        }
    }
    
    for model_name, config in models_config.items():
        logger.info(f"Benchmarking {model_name}...")
        
        try:
            processor_or_transform, model, is_vit = config['loader']()
            model.to(device)
            model.eval()
            
            if config['weights']:
                weights_path = hf_hub_download(repo_id=config['weights'], filename="best_model_run_eif1jakb.pth")
                state_dict = torch.load(weights_path, map_location=device)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
            
            times = []
            
            with torch.no_grad():
                for img_path in tqdm(sample_images, desc=f"Benchmark {model_name}"):
                    image = Image.open(img_path).convert('RGB')
                    
                    start_time = time.time()
                    
                    if is_vit:
                        inputs = processor_or_transform(images=image, return_tensors="pt")
                        pixel_values = inputs['pixel_values'].to(device)
                        outputs = model(pixel_values)
                    else:
                        image_tensor = processor_or_transform(image).unsqueeze(0).to(device)
                        outputs = model(image_tensor)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            benchmark_results[model_name] = {
                'mean_time_ms': float(np.mean(times) * 1000),
                'std_time_ms': float(np.std(times) * 1000),
                'min_time_ms': float(np.min(times) * 1000),
                'max_time_ms': float(np.max(times) * 1000),
                'median_time_ms': float(np.median(times) * 1000),
                'fps': float(1.0 / np.mean(times)),
                'total_samples': len(times)
            }
            
            logger.info(f"{model_name}: {benchmark_results[model_name]['mean_time_ms']:.2f} ms/image, "
                       f"{benchmark_results[model_name]['fps']:.2f} FPS")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            continue
    
    with open(results_dir / 'inference_time_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    benchmark_df = pd.DataFrame([
        {
            'model': name,
            'mean_ms': stats['mean_time_ms'],
            'fps': stats['fps'],
            'std_ms': stats['std_time_ms']
        }
        for name, stats in benchmark_results.items()
    ])
    benchmark_df = benchmark_df.sort_values('mean_ms')
    benchmark_df.to_csv(results_dir / 'inference_time_comparison.csv', index=False)
    
    logger.info("Inference time benchmarking completed")
    return benchmark_results


def generate_summary_statistics(results_dir):
    """Generate comprehensive summary statistics"""
    logger.info("Generating final summary statistics...")
    
    results_dir = Path(results_dir)
    
    comparison_df = pd.read_csv(results_dir / 'model_comparison.csv')
    
    summary = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'total_models_evaluated': len(comparison_df),
        'best_performers': {
            'highest_auc': {
                'model': comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'model_name'],
                'value': float(comparison_df['roc_auc'].max())
            },
            'lowest_eer': {
                'model': comparison_df.loc[comparison_df['eer'].idxmin(), 'model_name'],
                'value': float(comparison_df['eer'].min())
            },
            'highest_accuracy': {
                'model': comparison_df.loc[comparison_df['t0.5_accuracy'].idxmax(), 'model_name'],
                'value': float(comparison_df['t0.5_accuracy'].max())
            }
        },
        'performance_ranges': {
            'auc': {'min': float(comparison_df['roc_auc'].min()), 'max': float(comparison_df['roc_auc'].max())},
            'eer': {'min': float(comparison_df['eer'].min()), 'max': float(comparison_df['eer'].max())},
            'accuracy': {'min': float(comparison_df['t0.5_accuracy'].min()), 'max': float(comparison_df['t0.5_accuracy'].max())}
        }
    }
    
    with open(results_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Summary statistics generated")
    return summary


def main():
    results_dir = Path('results_inf_final_analysis')
    test_data_dir = Path('test_inf')
    
    logger.info("="*80)
    logger.info("ADDITIONAL ANALYSIS PIPELINE")
    logger.info("="*80)
    
    logger.info("\n[1/4] Analyzing score distributions...")
    distributions = analyze_score_distributions(results_dir)
    
    logger.info("\n[2/4] Extracting failed cases...")
    failed_cases = extract_failed_cases(results_dir, test_data_dir, top_n=20)
    
    logger.info("\n[3/4] Benchmarking inference time...")
    benchmark = benchmark_inference_time(results_dir, test_data_dir, num_samples=100)
    
    logger.info("\n[4/4] Generating summary statistics...")
    summary = generate_summary_statistics(results_dir)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nGenerated files in {results_dir}/:")
    logger.info("  - score_distribution_analysis.json")
    logger.info("  - score_separation_comparison.csv")
    logger.info("  - inference_time_benchmark.json")
    logger.info("  - inference_time_comparison.csv")
    logger.info("  - failed_cases_analysis/ (with images)")
    logger.info("  - final_summary.json")
    
    logger.info("\nBest Model by AUC: " + summary['best_performers']['highest_auc']['model'])
    logger.info(f"  AUC: {summary['best_performers']['highest_auc']['value']:.4f}")


if __name__ == "__main__":
    main()
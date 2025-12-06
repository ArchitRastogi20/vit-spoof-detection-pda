import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import timm
from PIL import Image
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import warnings
import json
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
class Config:
    # Dataset
    data_root = "./augmented_images"
    train_split = 0.85
    val_split = 0.15
    
    # Model
    model_name = "vit_base_patch16_224"
    pretrained = True
    num_classes = 2
    
    # Training - will be overridden by sweep
    batch_size = 128
    num_epochs = 50
    learning_rate = 3e-4
    weight_decay = 0.05
    warmup_epochs = 3
    dropout = 0.1
    
    # Loss function
    loss_type = "focal"  # "ce", "focal", "weighted_ce"
    focal_alpha = 0.25
    focal_gamma = 2.0
    
    # Optimization
    num_workers = 28
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4
    mixed_precision = True
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    
    # Augmentation
    img_size = 224
    random_erase_prob = 0.25
    
    # Scheduler
    scheduler_type = "cosine"
    min_lr = 1e-6
    
    # Early stopping
    early_stopping_patience = 10
    early_stopping_min_delta = 0.001
    
    # Checkpointing
    save_dir = "./checkpoints_advanced"
    log_interval = 10
    
    # Threshold optimization
    optimize_threshold = True
    threshold_min = 0.3
    threshold_max = 0.7
    threshold_steps = 41
    
    # Wandb
    wandb_project = "face-antispoofing-advanced-v2-simple"
    wandb_entity = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================== DATASET ====================
class AugmentedDataset(Dataset):
    def __init__(self, file_list, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        for item in file_list:
            self.samples.append(str(self.data_root / item['path']))
            self.labels.append(item['label'])
        
        logger.info(f"Loaded {len(self.samples)} images - Live: {sum(self.labels)}, Spoof: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def scan_augmented_dataset(data_root):
    """Scan augmented dataset"""
    data_root = Path(data_root)
    file_list = []
    
    logger.info(f"Scanning augmented dataset at {data_root}")
    
    live_dir = data_root / "live"
    spoof_dir = data_root / "spoof"
    
    if live_dir.exists():
        for img_path in live_dir.glob("*.jpg"):
            rel_path = img_path.relative_to(data_root)
            file_list.append({'path': str(rel_path), 'label': 1})
    
    if spoof_dir.exists():
        for img_path in spoof_dir.glob("*.jpg"):
            rel_path = img_path.relative_to(data_root)
            file_list.append({'path': str(rel_path), 'label': 0})
    
    logger.info(f"Found {len(file_list)} images")
    return file_list


def get_transforms(config, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=config.random_erase_prob),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ==================== MODEL ====================
class ViTFaceAntiSpoofing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=0)
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


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ==================== THRESHOLD OPTIMIZATION ====================
def find_optimal_threshold(labels, probs, config):
    """Find optimal decision threshold"""
    thresholds = np.linspace(config.threshold_min, config.threshold_max, config.threshold_steps)
    
    best_threshold = 0.5
    best_f1 = 0
    best_acc = 0
    
    results = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_acc = acc
    
    # Log threshold sweep to wandb
    for res in results:
        wandb.log({
            'threshold_sweep/threshold': res['threshold'],
            'threshold_sweep/accuracy': res['accuracy'],
            'threshold_sweep/precision': res['precision'],
            'threshold_sweep/recall': res['recall'],
            'threshold_sweep/f1': res['f1'],
        })
    
    logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f}, Acc: {best_acc:.4f})")
    return best_threshold, best_f1, best_acc


# ==================== TRAINING UTILITIES ====================
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss_function(config, class_weights=None):
    """Get loss function based on config"""
    if config.loss_type == "focal":
        logger.info(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    elif config.loss_type == "weighted_ce":
        logger.info(f"Using Weighted CrossEntropy (weights={class_weights})")
        if class_weights is not None:
            weights = torch.tensor(class_weights).to(config.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()
    else:
        logger.info("Using standard CrossEntropy")
        return nn.CrossEntropyLoss()


def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, config, epoch, global_step):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [TRAIN]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        with autocast(enabled=config.mixed_precision):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
        
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accs.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        if batch_idx % config.log_interval == 0:
            wandb.log({
                'train/loss': losses.val,
                'train/acc': accs.val,
                'train/loss_avg': losses.avg,
                'train/acc_avg': accs.avg,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch + batch_idx / len(loader),
                'train/step': global_step,
            })
    
    return losses.avg, accs.avg, global_step


@torch.no_grad()
def validate(model, loader, criterion, config, epoch, phase='val', optimize_threshold=False):
    model.eval()
    losses = AverageMeter()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [{phase.upper()}]")
    
    for images, labels in pbar:
        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        with autocast(enabled=config.mixed_precision):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        losses.update(loss.item(), images.size(0))
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Default threshold metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Log metrics with default threshold (0.5)
    wandb.log({
        f'{phase}/loss': losses.avg,
        f'{phase}/accuracy': accuracy,
        f'{phase}/precision': precision,
        f'{phase}/recall': recall,
        f'{phase}/f1': f1,
        f'{phase}/auc': auc,
        f'{phase}/specificity': specificity,
        f'{phase}/npv': npv,
        f'{phase}/tp': tp,
        f'{phase}/tn': tn,
        f'{phase}/fp': fp,
        f'{phase}/fn': fn,
        f'{phase}/far': far,
        f'{phase}/frr': frr,
        f'{phase}/epoch': epoch,
    })
    
    # Optimize threshold if requested
    optimal_threshold = 0.5
    optimal_f1 = f1
    optimal_acc = accuracy
    
    if optimize_threshold and phase == 'val':
        optimal_threshold, optimal_f1, optimal_acc = find_optimal_threshold(
            all_labels, all_probs, config
        )
        
        # Calculate metrics with optimal threshold
        optimal_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
        opt_precision, opt_recall, _, _ = precision_recall_fscore_support(
            all_labels, optimal_preds, average='binary', zero_division=0
        )
        opt_cm = confusion_matrix(all_labels, optimal_preds)
        opt_tn, opt_fp, opt_fn, opt_tp = opt_cm.ravel()
        opt_specificity = opt_tn / (opt_tn + opt_fp) if (opt_tn + opt_fp) > 0 else 0
        opt_far = opt_fp / (opt_fp + opt_tn) if (opt_fp + opt_tn) > 0 else 0
        opt_frr = opt_fn / (opt_fn + opt_tp) if (opt_fn + opt_tp) > 0 else 0
        
        wandb.log({
            f'{phase}/optimal_threshold': optimal_threshold,
            f'{phase}/optimal_accuracy': optimal_acc,
            f'{phase}/optimal_precision': opt_precision,
            f'{phase}/optimal_recall': opt_recall,
            f'{phase}/optimal_f1': optimal_f1,
            f'{phase}/optimal_specificity': opt_specificity,
            f'{phase}/optimal_far': opt_far,
            f'{phase}/optimal_frr': opt_frr,
            f'{phase}/optimal_tp': opt_tp,
            f'{phase}/optimal_tn': opt_tn,
            f'{phase}/optimal_fp': opt_fp,
            f'{phase}/optimal_fn': opt_fn,
        })
        
        logger.info(f"{phase.upper()} - Loss: {losses.avg:.4f}, AUC: {auc:.4f}")
        logger.info(f"  Default (0.5): Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
        logger.info(f"  Optimal ({optimal_threshold:.3f}): Acc: {optimal_acc:.4f}, F1: {optimal_f1:.4f}, Prec: {opt_precision:.4f}, Rec: {opt_recall:.4f}")
    else:
        logger.info(f"{phase.upper()} - Loss: {losses.avg:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, "
                    f"Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return losses.avg, optimal_acc, optimal_f1, auc, optimal_threshold


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'config': vars(config),
    }
    
    save_path = Path(config.save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def train():
    config = Config()
    
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config))
    
    # Update config from wandb sweep
    config.learning_rate = wandb.config.get('learning_rate', config.learning_rate)
    config.batch_size = wandb.config.get('batch_size', config.batch_size)
    config.weight_decay = wandb.config.get('weight_decay', config.weight_decay)
    config.dropout = wandb.config.get('dropout', config.dropout)
    config.loss_type = wandb.config.get('loss_type', config.loss_type)
    config.focal_alpha = wandb.config.get('focal_alpha', config.focal_alpha)
    config.focal_gamma = wandb.config.get('focal_gamma', config.focal_gamma)
    config.num_epochs = wandb.config.get('num_epochs', config.num_epochs)
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading augmented dataset...")
    file_list = scan_augmented_dataset(config.data_root)
    
    # Calculate class weights
    labels = [item['label'] for item in file_list]
    live_count = sum(labels)
    spoof_count = len(labels) - live_count
    total = len(labels)
    
    class_weights = [total / (2 * spoof_count), total / (2 * live_count)]
    logger.info(f"Class distribution - Live: {live_count}, Spoof: {spoof_count}")
    logger.info(f"Class weights - Spoof: {class_weights[0]:.4f}, Live: {class_weights[1]:.4f}")
    
    wandb.log({
        'dataset/live_count': live_count,
        'dataset/spoof_count': spoof_count,
        'dataset/ratio': spoof_count / live_count,
        'dataset/class_weight_spoof': class_weights[0],
        'dataset/class_weight_live': class_weights[1],
    })
    
    # Split dataset
    train_list, val_list = train_test_split(
        file_list,
        test_size=config.val_split,
        random_state=config.seed,
        stratify=labels
    )
    
    logger.info(f"Split - Train: {len(train_list)}, Val: {len(val_list)}")
    
    # Create datasets
    train_dataset = AugmentedDataset(train_list, config.data_root, get_transforms(config, is_train=True))
    val_dataset = AugmentedDataset(val_list, config.data_root, get_transforms(config, is_train=False))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    # Model
    logger.info("Creating model...")
    model = ViTFaceAntiSpoofing(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    wandb.log({
        'model/total_params': total_params,
        'model/trainable_params': trainable_params,
    })
    
    # Loss function
    criterion = get_loss_function(config, class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(num_training_steps * config.warmup_epochs / config.num_epochs)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=config.min_lr
    )
    
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode='max'
    )
    
    # Training loop
    logger.info("Starting training...")
    best_f1 = 0.0
    best_acc = 0.0
    best_auc = 0.0
    best_threshold = 0.5
    global_step = 0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs}\n{'='*60}")
        
        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, config, epoch, global_step
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, optimal_threshold = validate(
            model, val_loader, criterion, config, epoch, 'val',
            optimize_threshold=config.optimize_threshold
        )
        
        # Log epoch summary
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'train/epoch_acc': train_acc,
            'val/epoch_loss': val_loss,
            'val/epoch_acc': val_acc,
            'val/epoch_f1': val_f1,
            'val/epoch_auc': val_auc,
        })
        
        # Save best model
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_acc = val_acc
            best_auc = val_auc
            best_threshold = optimal_threshold
            logger.info(f"New best! F1: {best_f1:.4f}, Acc: {best_acc:.4f}, AUC: {best_auc:.4f}, Threshold: {best_threshold:.3f}")
            
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {'f1': best_f1, 'acc': best_acc, 'auc': best_auc, 'threshold': best_threshold},
                config, f"best_model_run_{wandb.run.id}.pth"
            )
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {'f1': val_f1, 'acc': val_acc, 'auc': val_auc, 'threshold': optimal_threshold},
                config, f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        # Early stopping check
        if early_stopping(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            wandb.log({'early_stopped': True, 'stopped_epoch': epoch})
            break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Log final metrics
    wandb.log({
        'final/best_f1': best_f1,
        'final/best_acc': best_acc,
        'final/best_auc': best_auc,
        'final/best_threshold': best_threshold,
    })
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best - F1: {best_f1:.4f}, Acc: {best_acc:.4f}, AUC: {best_auc:.4f}, Threshold: {best_threshold:.3f}")
    wandb.finish()


# ==================== HYPERPARAMETER SWEEP ====================
def run_sweep():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val/optimal_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-5,
                'max': 5e-4
            },
            'batch_size': {
                'values': [96, 128, 160]
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 0.01,
                'max': 0.1
            },
            'dropout': {
                'values': [0.0, 0.1, 0.2]
            },
            'loss_type': {
                'values': ['focal', 'weighted_ce', 'ce']
            },
            'focal_alpha': {
                'values': [0.15, 0.25, 0.35]
            },
            'focal_gamma': {
                'values': [1.5, 2.0, 2.5]
            },
            'num_epochs': {
                'value': 50
            }
        }
    }
    
    config = Config()
    sweep_id = wandb.sweep(sweep_config, project=config.wandb_project, entity=config.wandb_entity)
    
    logger.info(f"Sweep ID: {sweep_id}")
    logger.info("Starting hyperparameter sweep with 12 runs...")
    
    wandb.agent(sweep_id, function=train, count=12)
    
    logger.info("Sweep completed!")


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
        logger.info("Running hyperparameter sweep...")
        run_sweep()
    else:
        logger.info("Running single training run...")
        logger.info("For hyperparameter sweep, use: python train_advanced.py --sweep")
        train()
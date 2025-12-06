import os
import json
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import warnings
import shutil
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
class Config:
    # Dataset
    data_root = "./celeba_spoof"
    save_splits = True
    train_split = 0.85
    val_split = 0.15
    
    # Model
    model_name = "vit_base_patch16_224"
    pretrained = True
    num_classes = 2
    
    # Training - defaults
    batch_size = 128
    num_epochs = 30
    learning_rate = 3e-4
    weight_decay = 0.05
    warmup_epochs = 3
    label_smoothing = 0.1
    dropout = 0.1
    
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
    
    # Checkpointing
    save_dir = "./checkpoints"
    log_interval = 10
    
    # Wandb
    wandb_project = "face-antispoofing-vit-sweep"
    wandb_entity = None
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


# ==================== DATASET ====================
class CelebASpoofDataset(Dataset):
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


def scan_dataset(data_root):
    """Scan dataset and create file list"""
    data_root = Path(data_root)
    file_list = []
    
    logger.info(f"Scanning dataset at {data_root}")
    
    for subject_dir in sorted(data_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        live_dir = subject_dir / "live"
        spoof_dir = subject_dir / "spoof"
        
        if live_dir.exists():
            for img_path in live_dir.glob("*.png"):
                rel_path = img_path.relative_to(data_root)
                file_list.append({'path': str(rel_path), 'label': 1})
        
        if spoof_dir.exists():
            for img_path in spoof_dir.glob("*.png"):
                rel_path = img_path.relative_to(data_root)
                file_list.append({'path': str(rel_path), 'label': 0})
    
    logger.info(f"Found {len(file_list)} images")
    return file_list


def create_splits(file_list, train_split=0.85, val_split=0.15, seed=42):
    """Create train/val splits and save to txt files"""
    labels = [item['label'] for item in file_list]
    
    train_list, val_list = train_test_split(
        file_list,
        test_size=val_split,
        random_state=seed,
        stratify=labels
    )
    
    logger.info(f"Split - Train: {len(train_list)}, Val: {len(val_list)}")
    
    with open('train_files.txt', 'w') as f:
        for item in train_list:
            f.write(f"{item['path']}\t{item['label']}\n")
    
    with open('val_files.txt', 'w') as f:
        for item in val_list:
            f.write(f"{item['path']}\t{item['label']}\n")
    
    logger.info("Saved file lists to train_files.txt and val_files.txt")
    
    return train_list, val_list


def get_transforms(config, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
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


# ==================== TRAINING ====================
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
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{accs.avg:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
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
def validate(model, loader, criterion, config, epoch, phase='val'):
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    logger.info(f"{phase.upper()} - Loss: {losses.avg:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Spec: {specificity:.4f}")
    
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
        f'{phase}/epoch': epoch,
    })
    
    return losses.avg, accuracy, f1, auc


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


def train(config_dict=None):
    config = Config()
    
    # Initialize wandb with config
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=vars(config)
    )
    
    # Update config from wandb (if running sweep)
    if config_dict:
        for key, value in config_dict.items():
            setattr(config, key, value)
    else:
        # Update from wandb.config for sweep
        config.learning_rate = wandb.config.get('learning_rate', config.learning_rate)
        config.batch_size = wandb.config.get('batch_size', config.batch_size)
        config.weight_decay = wandb.config.get('weight_decay', config.weight_decay)
        config.dropout = wandb.config.get('dropout', config.dropout)
        config.label_smoothing = wandb.config.get('label_smoothing', config.label_smoothing)
        config.num_epochs = wandb.config.get('num_epochs', config.num_epochs)
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Scan and split dataset
    logger.info("Scanning dataset...")
    file_list = scan_dataset(config.data_root)
    
    train_list, val_list = create_splits(file_list, config.train_split, config.val_split, config.seed)
    
    # Create datasets
    train_dataset = CelebASpoofDataset(train_list, config.data_root, get_transforms(config, is_train=True))
    val_dataset = CelebASpoofDataset(val_list, config.data_root, get_transforms(config, is_train=False))
    
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
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    
    num_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(num_training_steps * config.warmup_epochs / config.num_epochs)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps, eta_min=config.min_lr)
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Training loop
    logger.info("Starting training...")
    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs}\n{'='*60}")
        
        train_loss, train_acc, global_step = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, config, epoch, global_step)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, config, epoch, 'val')
        
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'train/epoch_acc': train_acc,
            'val/epoch_loss': val_loss,
            'val/epoch_acc': val_acc,
            'val/epoch_f1': val_f1,
            'val/epoch_auc': val_auc,
        })
        
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_f1 = val_f1
            best_auc = val_auc
            logger.info(f"New best! Acc: {best_acc:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
            
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {'acc': best_acc, 'f1': best_f1, 'auc': best_auc},
                config, f"best_model_run_{wandb.run.id}.pth"
            )
        
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {'acc': val_acc, 'f1': val_f1, 'auc': val_auc},
                config, f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        gc.collect()
        torch.cuda.empty_cache()
    
    wandb.log({
        'final/best_acc': best_acc,
        'final/best_f1': best_f1,
        'final/best_auc': best_auc,
    })
    
    logger.info(f"\nTraining complete! Best - Acc: {best_acc:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
    wandb.finish()


# ==================== HYPERPARAMETER SWEEP ====================
def run_sweep():
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val/auc',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'batch_size': {
                'values': [64, 96, 128]
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 0.001,
                'max': 0.1
            },
            'dropout': {
                'values': [0.0, 0.1, 0.2]
            },
            'label_smoothing': {
                'values': [0.0, 0.05, 0.1]
            },
            'num_epochs': {
                'value': 30
            }
        }
    }
    
    config = Config()
    sweep_id = wandb.sweep(sweep_config, project=config.wandb_project, entity=config.wandb_entity)
    
    logger.info(f"Sweep ID: {sweep_id}")
    logger.info("Starting hyperparameter sweep with 10 runs...")
    
    wandb.agent(sweep_id, function=train, count=10)
    
    logger.info("Sweep completed!")


# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
        logger.info("Running hyperparameter sweep...")
        run_sweep()
    else:
        logger.info("Running single training run...")
        logger.info("For hyperparameter sweep, use: python train.py --sweep")
        train()
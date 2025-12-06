import os
import sys
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import psutil
import resource
import warnings
warnings.filterwarnings('ignore')

# Set resource limits to 95%
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (int(62 * 0.95 * 1024 * 1024 * 1024), hard))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
class AugmentConfig:
    # Paths
    input_dir = "./celeba_spoof"
    output_dir = "./augmented_images"
    
    # Augmentation multipliers
    live_augmentations = 8
    spoof_augmentations = 2
    
    # Processing
    batch_size = 64
    num_workers = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image settings
    img_size = 224
    save_quality = 95
    
    # Memory management
    max_queue_size = 500
    gpu_memory_fraction = 0.95


# ==================== GPU-ACCELERATED AUGMENTATIONS ====================
class GPUAugmenter:
    """GPU-accelerated augmentation pipeline using Kornia"""
    
    def __init__(self, device='cuda'):
        self.device = device
        try:
            import kornia.augmentation as K
            self.use_kornia = True
            
            self.heavy_aug = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=20.0, p=0.7),
                K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
                K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5),
                K.RandomGaussianNoise(mean=0., std=0.05, p=0.3),
                K.RandomPerspective(distortion_scale=0.2, p=0.4),
                K.RandomElasticTransform(p=0.3),
                K.RandomSharpness(sharpness=2.0, p=0.3),
            ).to(device)
            
            self.medium_aug = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15.0, p=0.6),
                K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.4),
                K.RandomGaussianNoise(mean=0., std=0.03, p=0.2),
                K.RandomPerspective(distortion_scale=0.15, p=0.3),
            ).to(device)
            
            self.light_aug = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=10.0, p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.3),
            ).to(device)
            
            logger.info("Kornia GPU augmentation enabled")
        except ImportError:
            logger.warning("Kornia not found, using CPU augmentation")
            self.use_kornia = False
    
    @torch.no_grad()
    def augment_batch_gpu(self, images_tensor, aug_level='heavy'):
        """Augment batch of images on GPU"""
        if not self.use_kornia:
            return None
        
        if aug_level == 'heavy':
            aug_pipeline = self.heavy_aug
        elif aug_level == 'medium':
            aug_pipeline = self.medium_aug
        else:
            aug_pipeline = self.light_aug
        
        augmented = aug_pipeline(images_tensor)
        return augmented


# ==================== DATASET ====================
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [str(p) for p in image_paths]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            black_img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                return self.transform(black_img), img_path
            return black_img, img_path


# ==================== PROCESSING FUNCTIONS ====================
def scan_images(input_dir):
    """Scan input directory for images"""
    input_path = Path(input_dir)
    
    live_images = []
    spoof_images = []
    
    logger.info(f"Scanning directory: {input_dir}")
    
    for subject_dir in sorted(input_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        live_dir = subject_dir / "live"
        if live_dir.exists():
            live_images.extend(list(live_dir.glob("*.png")))
        
        spoof_dir = subject_dir / "spoof"
        if spoof_dir.exists():
            spoof_images.extend(list(spoof_dir.glob("*.png")))
    
    logger.info(f"Found {len(live_images)} live images and {len(spoof_images)} spoof images")
    return live_images, spoof_images


def save_image_tensor(tensor, save_path, quality=95):
    """Save tensor as image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU for processing
    tensor = tensor.cpu()
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp and convert to PIL
    tensor = torch.clamp(tensor, 0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(save_path, quality=quality, optimize=True)


def process_batch_gpu(images, image_paths, augmenter, output_dir, class_name, num_augmentations, config):
    """Process a batch of images on GPU"""
    images = images.to(config.device)
    
    saved_count = 0
    
    for aug_idx in range(num_augmentations):
        if aug_idx < num_augmentations // 3:
            aug_level = 'heavy'
        elif aug_idx < 2 * num_augmentations // 3:
            aug_level = 'medium'
        else:
            aug_level = 'light'
        
        augmented = augmenter.augment_batch_gpu(images, aug_level=aug_level)
        
        if augmented is None:
            continue
        
        for i, (aug_img, orig_path) in enumerate(zip(augmented, image_paths)):
            orig_filename = Path(orig_path).stem
            output_filename = f"{orig_filename}_aug{aug_idx}.jpg"
            output_path = Path(output_dir) / class_name / output_filename
            
            try:
                save_image_tensor(aug_img, output_path, quality=config.save_quality)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving {output_path}: {e}")
    
    return saved_count


def process_class(image_paths, output_dir, class_name, num_augmentations, config):
    """Process all images of a class"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {class_name} images: {len(image_paths)} images")
    logger.info(f"Generating {num_augmentations} augmentations per image")
    logger.info(f"Total output: {len(image_paths) * num_augmentations} images")
    logger.info(f"{'='*60}\n")
    
    (Path(output_dir) / class_name).mkdir(parents=True, exist_ok=True)
    
    augmenter = GPUAugmenter(device=config.device)
    
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    
    total_saved = 0
    pbar = tqdm(dataloader, desc=f"Augmenting {class_name}")
    
    for images, paths in pbar:
        saved = process_batch_gpu(images, paths, augmenter, output_dir, class_name, num_augmentations, config)
        total_saved += saved
        
        pbar.set_postfix({
            'saved': total_saved,
            'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    logger.info(f"Completed {class_name}: {total_saved} augmented images saved")
    return total_saved


def copy_original_images(image_paths, output_dir, class_name):
    """Copy original images to output directory"""
    logger.info(f"Copying {len(image_paths)} original {class_name} images...")
    
    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for img_path in tqdm(image_paths, desc=f"Copying {class_name}"):
        try:
            img = Image.open(img_path).convert('RGB')
            output_file = output_path / f"{img_path.stem}_orig.jpg"
            img.save(output_file, quality=95, optimize=True)
            copied += 1
        except Exception as e:
            logger.warning(f"Error copying {img_path}: {e}")
    
    logger.info(f"Copied {copied}/{len(image_paths)} original {class_name} images")
    return copied


def print_statistics(config, live_images, spoof_images, live_saved, spoof_saved):
    """Print augmentation statistics"""
    print("\n" + "="*60)
    print("AUGMENTATION STATISTICS")
    print("="*60)
    print(f"\nOriginal Dataset:")
    print(f"  Live images:  {len(live_images):,}")
    print(f"  Spoof images: {len(spoof_images):,}")
    print(f"  Ratio:        1:{len(spoof_images)/len(live_images):.2f} (spoof:live)")
    
    print(f"\nAugmentation Settings:")
    print(f"  Live augmentations:  {config.live_augmentations}x")
    print(f"  Spoof augmentations: {config.spoof_augmentations}x")
    
    print(f"\nAugmented Dataset:")
    print(f"  Live images:  {live_saved:,}")
    print(f"  Spoof images: {spoof_saved:,}")
    print(f"  Ratio:        1:{spoof_saved/live_saved:.2f} (spoof:live)")
    
    print(f"\nStorage:")
    output_path = Path(config.output_dir)
    if output_path.exists():
        output_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        print(f"  Total size:   {output_size/1024**3:.2f} GB")
    
    print(f"\nOutput directory: {config.output_dir}")
    print("="*60 + "\n")


# ==================== MAIN ====================
def main():
    config = AugmentConfig()
    
    logger.info("="*60)
    logger.info("FACE ANTI-SPOOFING DATA AUGMENTATION")
    logger.info("="*60)
    logger.info(f"Input:  {config.input_dir}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU:    {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
    logger.info(f"CPU:    {config.num_workers} workers")
    logger.info(f"RAM:    {psutil.virtual_memory().total/1024**3:.1f} GB")
    logger.info("="*60 + "\n")
    
    live_images, spoof_images = scan_images(config.input_dir)
    
    if len(live_images) == 0 or len(spoof_images) == 0:
        logger.error("No images found! Check input directory.")
        return
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    live_orig = copy_original_images(live_images, config.output_dir, 'live')
    spoof_orig = copy_original_images(spoof_images, config.output_dir, 'spoof')
    
    live_aug = process_class(
        live_images,
        config.output_dir,
        'live',
        config.live_augmentations,
        config
    )
    
    spoof_aug = process_class(
        spoof_images,
        config.output_dir,
        'spoof',
        config.spoof_augmentations,
        config
    )
    
    live_total = live_orig + live_aug
    spoof_total = spoof_orig + spoof_aug
    
    print_statistics(config, live_images, spoof_images, live_total, spoof_total)
    
    logger.info("Data augmentation completed successfully!")


if __name__ == "__main__":
    try:
        import kornia
        logger.info("Kornia found - GPU acceleration enabled")
    except ImportError:
        logger.warning("Kornia not found - installing...")
        os.system("pip install --break-system-packages kornia -q")
        logger.info("Kornia installed")
    
    main()
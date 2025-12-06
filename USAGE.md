# Usage Guide

## Dataset Preparation

### CelebA-Spoof Dataset

This project uses the CelebA-Spoof dataset. The official dataset is available at:
https://github.com/ZhangYuanhan-AI/CelebA-Spoof

### Download Issues and Workaround

The official repository provides two download options:
- Google Drive (requires manual download of individual shards)
- Baidu Drive (may not be accessible in all regions)

**Known Issues:**
- The Google Drive link does not support direct `gdown` download due to the multi-part archive structure
- Baidu Drive may require VPN access depending on your location
- The dataset consists of 100 sharded files (`.zip.001` through `.zip.100`)

### Recommended Download Process

**Option 1: Manual Download (Most Reliable)**

1. Visit the official CelebA-Spoof GitHub repository
2. Access the Google Drive link provided
3. Manually download the required shards (`.zip.001` through `.zip.100`)
4. For this project, we used shards 001-022 due to storage constraints

**Option 2: Using Personal Google Drive Mirror**

If you've already downloaded the shards, you can host them on your own Google Drive and use `gdown`:

```bash
# Install gdown
pip install gdown

# Download from your Google Drive folder (replace YOUR_FOLDER_ID)
gdown --folder "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" -O ./celeba_spoof

# Verify download
ls -lh ./celeba_spoof
```

### Extracting the Dataset

The CelebA-Spoof dataset uses multi-part 7z archives. You need `p7zip` to extract:

```bash
# Install p7zip (Ubuntu/Debian)
apt-get update -y && apt-get install -y p7zip-full

# Navigate to download directory
cd celeba_spoof

# Extract the dataset (extracts all parts automatically)
7z x CelebA_Spoof.zip.001 -o../data

# The extraction will automatically process all .zip.00X files
```

**Note:** You only need to extract from `.zip.001` - the 7z tool will automatically process all subsequent parts.

### Verify Dataset Structure

After extraction, your dataset should have the following structure:

```
data/
├── CelebA_Spoof/
│   ├── Data/
│   │   ├── train/
│   │   ├── test/
│   │   └── ...
│   └── metas/
│       ├── intra_test/
│       │   └── train_label.json
│       └── ...
```

## Installation

### Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Or manually install core packages:

```bash
pip install torch torchvision transformers timm kornia scikit-learn pandas numpy seaborn matplotlib pillow wandb tqdm
```

## Data Augmentation

Generate augmented training data to address class imbalance:

```bash
python augment_data.py
```

**Configuration:**
Edit `augment_data.py` to adjust:
- `input_dir`: Path to extracted CelebA-Spoof dataset
- `output_dir`: Path for augmented images (default: `./augmented_images`)
- `live_augmentations`: Augmentation multiplier for live samples (default: 8)
- `spoof_augmentations`: Augmentation multiplier for spoof samples (default: 2)

This will generate a balanced dataset with GPU-accelerated augmentation using Kornia.

## Training

### Main Training (Advanced)

Train the model with Focal Loss and advanced techniques:

```bash
python train_advanced.py
```

**Key configurations in script:**
- `data_root`: Path to augmented dataset
- `batch_size`: Training batch size (default: 128)
- `num_epochs`: Maximum training epochs (default: 50)
- `learning_rate`: Initial learning rate (default: 3e-4)
- `loss_type`: Loss function ("focal", "weighted_ce", or "ce")

The training script includes:
- Automatic mixed precision (AMP)
- Early stopping
- Checkpoint saving
- WandB logging (optional)
- Threshold optimization

### Hyperparameter Sweep (Optional)

Run Bayesian hyperparameter optimization:

```bash
python train_advanced.py --sweep
```

This will run 12 training experiments with different hyperparameter combinations.

### Baseline Training

Train the baseline model without advanced techniques:

```bash
cd simple
python train.py
```

## Testing

### Evaluate Trained Model

Run evaluation on test set:

```bash
python test.py
```

**Configuration:**
Edit `test.py` to adjust:
- `test_data_root`: Path to test split
- `checkpoint_path`: Path to trained model checkpoint
- `output_dir`: Directory for results (default: `./test_results`)

**Outputs:**
- Confusion matrix (PNG and CSV)
- ROC curve
- Per-image predictions
- Per-subject accuracy
- Comprehensive metrics (APCER, BPCER, EER, AUC-ROC)

## Visualization

### APCER vs BPCER Trade-off Curve

Generate threshold analysis visualization:

```bash
python data_vis.py
```

**Configuration:**
Edit `data_vis.py` to set:
- `FILE_PATH`: Path to per-image results CSV
- `OUTPUT_PATH`: Output path for visualization

This generates:
- APCER vs BPCER curve
- EER point identification
- Default threshold (0.5) comparison
- Metrics CSV with all thresholds

## Pre-trained Model

The trained model is available on Hugging Face:
https://huggingface.co/ArchitRastogi/vit-spoof-detection-pda

Download and use in your code:

```python
from transformers import AutoModel
import torch

# Load model
model = AutoModel.from_pretrained("ArchitRastogi/vit-spoof-detection-pda", trust_remote_code=True)
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
```


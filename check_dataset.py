"""
Quick script to check dataset output shapes and labels
"""

import torch
from config import ProjectConfig
from data_loader import create_dataloaders

cfg = ProjectConfig()
cfg.model.use_2d_conv = True

print("Creating dataloader...")
train_loader, val_loader, test_loader = create_dataloaders(
    project_cfg=cfg,
    batch_size=4,
    root="./data",
    hello_dir="data/hello",
    combine_custom_with_sc=True,
)

print("\n" + "="*60)
print("Checking Training Data")
print("="*60)

# Check a few batches
for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Features shape: {features.shape}")
    print(f"  Features dtype: {features.dtype}")
    print(f"  Features stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Label distribution: {torch.bincount(labels, minlength=len(cfg.labels)).tolist()}")
    
    # Check expected shape for 2D model
    if cfg.model.use_2d_conv:
        expected_shape = (features.shape[0], 1, cfg.audio.n_mfcc, features.shape[2] if features.dim() == 3 else None)
        print(f"  Expected for 2D model: (batch, 1, n_mfcc={cfg.audio.n_mfcc}, time_frames)")
        if features.dim() == 3:
            # Need to add channel dimension
            features_2d = features.unsqueeze(1)
            print(f"  After unsqueeze(1): {features_2d.shape}")
    
    if batch_idx >= 2:
        break

print("\n" + "="*60)
print("Label Distribution Summary")
print("="*60)

all_labels = []
for features, labels in train_loader:
    all_labels.extend(labels.tolist())
    if len(all_labels) > 1000:
        break

all_labels = torch.tensor(all_labels)
label_counts = torch.bincount(all_labels, minlength=len(cfg.labels))
print(f"\nTotal samples checked: {len(all_labels)}")
for i, label in enumerate(cfg.labels):
    count = label_counts[i].item()
    percentage = (count / len(all_labels)) * 100
    print(f"  {label}: {count} ({percentage:.1f}%)")

print("\n" + "="*60)


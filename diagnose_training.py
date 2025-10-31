"""
Quick diagnostic script to identify training issues
"""

import torch
from config import ProjectConfig
from data_loader import create_dataloaders
from model_2d import build_model

cfg = ProjectConfig()
cfg.model.use_2d_conv = True

print("="*60)
print("Training Diagnostics")
print("="*60)

# Create small dataloader
train_loader, val_loader, test_loader = create_dataloaders(
    project_cfg=cfg,
    batch_size=8,
    hello_dir="data/hello",
    combine_custom_with_sc=True,
)

print("\n1. Checking dataset output shapes...")
sample_features, sample_labels = next(iter(train_loader))
print(f"   Batch features shape: {sample_features.shape}")
print(f"   Batch labels shape: {sample_labels.shape}")
print(f"   Labels in batch: {sample_labels.tolist()}")
print(f"   Label distribution: {torch.bincount(sample_labels, minlength=3).tolist()}")

# Check what shape model expects
print("\n2. Building model...")
model = build_model(cfg.model)
print(f"   Model type: {cfg.model.model_type}, 2D: {cfg.model.use_2d_conv}")

# Prepare features for model
print("\n3. Testing model forward pass...")
if cfg.model.use_2d_conv:
    # Should be (batch, 1, n_mfcc, time_frames)
    if sample_features.dim() == 3:
        test_features = sample_features.unsqueeze(1)  # (batch, 1, n_mfcc, time_frames)
    else:
        test_features = sample_features
else:
    test_features = sample_features

print(f"   Input to model shape: {test_features.shape}")
print(f"   Expected: (batch, 1, n_mfcc={cfg.audio.n_mfcc}, time_frames)")

try:
    model.eval()
    with torch.no_grad():
        output = model(test_features)
        print(f"   Model output shape: {output.shape}")
        print(f"   Expected: (batch, {cfg.model.num_classes})")
        print(f"   Sample output: {output[0]}")
        
        probs = torch.softmax(output, dim=1)
        print(f"   Sample probabilities: {probs[0]}")
        
        preds = torch.argmax(output, dim=1)
        print(f"   Predictions: {preds.tolist()}")
        print(f"   True labels: {sample_labels.tolist()}")
        print(f"   Correct: {(preds == sample_labels).sum().item()}/{len(sample_labels)}")
        
        # Check per-class predictions
        print(f"\n4. Per-class analysis:")
        for i, label_name in enumerate(cfg.labels):
            true_mask = (sample_labels == i)
            if true_mask.any():
                pred_mask = (preds == i)
                print(f"   {label_name}: {true_mask.sum().item()} true, {pred_mask.sum().item()} predicted")
except Exception as e:
    print(f"   ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Diagnostics complete")
print("="*60)


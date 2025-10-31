"""
Enhanced training script with 3-class support and Speech Commands dataset
Uses config-based architecture inspired by Hello repository
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import argparse
import numpy as np

from config import ProjectConfig
from model_2d import build_model, get_model_size_mb, count_parameters
from data_loader import create_dataloaders
from metrics import evaluate_model


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy"""
    predicted = preds.argmax(dim=1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0) if targets.size(0) > 0 else 0.0


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    cfg=None,
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for features, labels in dataloader:
        # Handle different input shapes for 2D model
        # Expected: (batch, 1, n_mfcc, time_frames) for 2D model
        # Or: (batch, 1, time) for 1D model
        
        if cfg.model.use_2d_conv:
            # 2D model expects: (batch, 1, n_mfcc, time_frames)
            # Dataset should return: (n_mfcc, time_frames) per sample
            # DataLoader batches it to: (batch, n_mfcc, time_frames)
            
            if features.dim() == 3:
                # Should be (batch, n_mfcc, time_frames)
                if features.shape[1] == cfg.audio.n_mfcc:
                    # Correct: (batch, n_mfcc, time_frames) -> (batch, 1, n_mfcc, time_frames)
                    features = features.unsqueeze(1)
                elif features.shape[2] == cfg.audio.n_mfcc:
                    # Might be (batch, time, n_mfcc) - transpose
                    features = features.transpose(1, 2).unsqueeze(1)
                else:
                    print(f"Error: Unexpected 3D shape for 2D model: {features.shape}, expected (batch, {cfg.audio.n_mfcc}, time_frames)")
                    # Try to fix: assume wrong dimension order
                    if features.shape[1] < features.shape[2] and features.shape[1] < cfg.audio.n_mfcc:
                        # Likely (batch, time, n_mfcc)
                        features = features.transpose(1, 2).unsqueeze(1)
            elif features.dim() == 4:
                # Already has channel dimension
                if features.shape[1] == 1:
                    # Correct: (batch, 1, n_mfcc, time_frames)
                    pass
                else:
                    print(f"Warning: 4D features with channel={features.shape[1]}, expected 1")
                    # Might need to rearrange
            elif features.dim() == 2:
                # This is wrong for 2D model - dataset should return 2D features
                print(f"ERROR: 2D model got 2D features (batch, time): {features.shape}")
                print(f"       Dataset should return (n_mfcc, time_frames) per sample")
                # Try to expand: (batch, time) -> (batch, 1, 1, time) - won't work well
                features = features.unsqueeze(1).unsqueeze(1)
        else:
            # 1D model expects: (batch, 1, time)
            if features.dim() == 2:
                features = features.unsqueeze(1)
            elif features.dim() == 3:
                # Might be (batch, n_mfcc, time) -> take first MFCC or flatten
                if features.shape[1] > 1:
                    features = features[:, 0:1, :]  # Take first MFCC coefficient
        
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Debug: Print shapes and predictions on first batch
        if cfg and not hasattr(train_one_epoch, '_first_batch_printed'):
            train_one_epoch._first_batch_printed = True
            print(f"\n{'='*60}")
            print("DEBUG - First Batch Analysis")
            print(f"{'='*60}")
            print(f"Features shape: {features.shape}")
            print(f"Expected for 2D model: (batch, 1, n_mfcc={cfg.audio.n_mfcc}, time_frames)")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels in batch: {labels.tolist()}")
            print(f"Label distribution: {torch.bincount(labels, minlength=len(cfg.labels)).tolist()}")
        
        logits = model(features)
        
        # Check logits and predictions
        if cfg and hasattr(train_one_epoch, '_first_batch_printed') and train_one_epoch._first_batch_printed:
            print(f"\nModel output:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Expected: (batch, {cfg.model.num_classes})")
            print(f"  Logits sample: {logits[0].cpu().tolist()}")
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            print(f"  Probabilities sample: {probs[0].cpu().tolist()}")
            print(f"  Predictions: {preds.cpu().tolist()}")
            print(f"  True labels: {labels.cpu().tolist()}")
            
            # Per-class breakdown
            print(f"\n  Per-class:")
            for i, label_name in enumerate(cfg.labels):
                true_count = (labels == i).sum().item()
                pred_count = (preds == i).sum().item()
                correct = ((preds == i) & (labels == i)).sum().item()
                print(f"    {label_name}: {true_count} true, {pred_count} predicted, {correct} correct")
            
            print(f"{'='*60}\n")
            train_one_epoch._first_batch_printed = False  # Reset so it only prints once
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        running_acc += accuracy(logits.detach(), labels) * features.size(0)
    
    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    cfg=None,
):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.inference_mode():
        for features, labels in dataloader:
            # Handle different input shapes (same as train_one_epoch)
            if cfg and cfg.model.use_2d_conv:
                # 2D model expects: (batch, 1, n_mfcc, time_frames)
                if features.dim() == 3:
                    # (batch, n_mfcc, time_frames) -> (batch, 1, n_mfcc, time_frames)
                    features = features.unsqueeze(1)
                elif features.dim() == 2:
                    features = features.unsqueeze(1).unsqueeze(1)
            else:
                # 1D model expects: (batch, 1, time)
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                elif features.dim() == 3:
                    if features.shape[1] > 1:
                        features = features[:, 0:1, :]
            
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            total_acc += accuracy(logits, labels) * features.size(0)
    
    dataset_size = len(dataloader.dataset)
    return total_loss / dataset_size, total_acc / dataset_size


def export_models(model, project_cfg: ProjectConfig, example_input, device):
    """Export models to various formats"""
    model = model.to("cpu").eval()
    
    export_paths = {}
    
    # 1. Save PyTorch checkpoint
    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "hello_keyword.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': project_cfg.model.model_type,
        'use_2d_conv': project_cfg.model.use_2d_conv,
        'num_classes': project_cfg.model.num_classes,
        'config': project_cfg,
    }, checkpoint_path)
    export_paths['pytorch'] = str(checkpoint_path)
    print(f"✓ Saved PyTorch checkpoint: {checkpoint_path}")
    
    # 2. Export TorchScript
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(model, example_input)
        torchscript_path = save_dir / project_cfg.model.torchscript_model_path.split("/")[-1]
        traced.save(str(torchscript_path))
        export_paths['torchscript'] = str(torchscript_path)
        torchscript_size = torchscript_path.stat().st_size / 1024
        print(f"✓ TorchScript model: {torchscript_path} ({torchscript_size:.1f} KB)")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
    
    # 3. Export ExecuTorch
    try:
        exported = torch.export.export(model, (example_input,))
        from executorch.exir import to_executorch
        
        executorch_program = to_executorch(exported)
        executorch_path = save_dir / project_cfg.model.exported_model_path.split("/")[-1]
        executorch_program.save(str(executorch_path))
        export_paths['executorch'] = str(executorch_path)
        executorch_size = executorch_path.stat().st_size / 1024
        print(f"✓ ExecuTorch model: {executorch_path} ({executorch_size:.1f} KB)")
    except ImportError:
        print("✗ ExecuTorch not installed. Install with: pip install executorch")
    except Exception as e:
        print(f"✗ ExecuTorch export failed: {e}")
    
    return export_paths


def train_keyword_spotter():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train keyword spotting model")
    parser.add_argument("--hello_dir", type=str, default=None,
                       help="Directory with hello audio files")
    parser.add_argument("--other_dir", type=str, default=None,
                       help="Directory with other audio files")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Root directory for Speech Commands dataset")
    parser.add_argument("--use_speech_commands", action="store_true",
                       help="Use Speech Commands dataset (auto-downloads)")
    parser.add_argument("--no_speech_commands", action="store_true",
                       help="Disable Speech Commands dataset")
    parser.add_argument("--combine", action="store_true",
                       help="Combine custom hello samples with Speech Commands (best of both!)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (uses config if not specified)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (uses config if not specified)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (uses config if not specified)")
    parser.add_argument("--model_type", type=str, choices=["tiny", "minimal"], default=None,
                       help="Model type")
    parser.add_argument("--use_2d", action="store_true",
                       help="Use 2D CNN model")
    parser.add_argument("--output", type=str, default="models/hello_model.pth",
                       help="Output model path")
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = ProjectConfig()
    
    # Override config from args
    if args.model_type:
        cfg.model.model_type = args.model_type
    if args.use_2d:
        cfg.model.use_2d_conv = True
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.use_speech_commands:
        cfg.training.use_speech_commands = True
    if args.no_speech_commands:
        cfg.training.use_speech_commands = False
    
    # Set random seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    print(f"{'='*60}")
    print("Keyword Spotting Training")
    print(f"{'='*60}")
    print(f"Model type: {cfg.model.model_type}")
    print(f"Use 2D CNN: {cfg.model.use_2d_conv}")
    print(f"Num classes: {cfg.model.num_classes} ({cfg.labels})")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    combine_mode = args.combine or (args.hello_dir and cfg.training.use_speech_commands and not args.no_speech_commands)
    if combine_mode:
        print(f"Dataset mode: COMBINED (custom hello + Speech Commands unknown)")
    else:
        print(f"Use Speech Commands: {cfg.training.use_speech_commands}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    try:
        # Determine if we should combine
        combine_mode = args.combine or (args.hello_dir and cfg.training.use_speech_commands and not args.no_speech_commands)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            project_cfg=cfg,
            batch_size=cfg.training.batch_size,
            root=args.data_root,
            hello_dir=args.hello_dir,
            other_dir=args.other_dir,
            use_speech_commands=cfg.training.use_speech_commands if not combine_mode else False,
            combine_custom_with_sc=combine_mode,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = build_model(cfg.model).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB\n")
    
    # Calculate class weights for imbalance
    # Count samples per class in training set (sample first few batches)
    print("\nAnalyzing class distribution...")
    class_counts = {}
    sample_limit = 1000  # Sample up to 1000 samples for speed
    samples_checked = 0
    
    for features, labels_batch in train_loader:
        for label in labels_batch.cpu().numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
            samples_checked += 1
            if samples_checked >= sample_limit:
                break
        if samples_checked >= sample_limit:
            break
    
    if len(class_counts) > 1:
        # Calculate weights: inverse frequency with sqrt smoothing for stability
        total_samples = sum(class_counts.values())
        max_count = max(class_counts.values())
        # Use sqrt of inverse frequency to reduce extreme weights
        # This gives stronger weighting without being too aggressive
        class_weights = torch.tensor([
            np.sqrt(max_count / max(class_counts.get(i, 1), 1)) for i in range(len(cfg.labels))
        ], dtype=torch.float32).to(device)
        
        # Apply additional boost for severely imbalanced classes (ratio > 3:1)
        for i, label in enumerate(cfg.labels):
            count = class_counts.get(i, 1)
            ratio = max_count / count
            if ratio > 3.0:
                # Boost weight for severely underrepresented classes
                class_weights[i] *= 1.5
        
        print(f"\nClass distribution (from {samples_checked} samples):")
        for i, label in enumerate(cfg.labels):
            count = class_counts.get(i, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            weight = class_weights[i].item()
            ratio = max_count / max(count, 1)
            print(f"  {label}: {count} ({percentage:.1f}%) - weight: {weight:.2f} (ratio: {ratio:.1f}x)")
        
        # Normalize weights to prevent extreme values
        class_weights = class_weights / class_weights.sum() * len(cfg.labels)
        print(f"\nUsing weighted loss to handle class imbalance (sqrt-scaled)")
    else:
        class_weights = None
        print(f"Warning: Could not determine class distribution properly")
    
    # Loss function
    if cfg.training.use_focal_loss:
        # Focal loss for better handling of hard negatives
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, weight=None):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.weight = weight
            
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(
            alpha=cfg.training.focal_alpha,
            gamma=cfg.training.focal_gamma,
            weight=class_weights
        )
        print("Using Weighted Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.label_smoothing,
            weight=class_weights
        )
        if class_weights is not None:
            print(f"Using Weighted CrossEntropyLoss with label smoothing={cfg.training.label_smoothing}")
        else:
            print(f"Using CrossEntropyLoss with label smoothing={cfg.training.label_smoothing}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3  # Increased patience to prevent premature LR drops
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10  # Stop if validation doesn't improve for 10 epochs
    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / args.output.split("/")[-1]
    
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}\n")
    
    for epoch in range(cfg.training.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg=cfg
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, cfg=cfg
        )
        scheduler.step(val_loss)
        
        # Debug: Show validation predictions breakdown
        if epoch < 3 or epoch % 10 == 0:
            with torch.inference_mode():
                all_val_preds = []
                all_val_labels = []
                for val_features, val_labels in val_loader:
                    if cfg.model.use_2d_conv:
                        if val_features.dim() == 3:
                            val_features = val_features.unsqueeze(1)
                    val_features = val_features.to(device)
                    val_labels = val_labels.to(device)
                    val_logits = model(val_features)
                    val_preds = val_logits.argmax(dim=1)
                    all_val_preds.extend(val_preds.cpu().tolist())
                    all_val_labels.extend(val_labels.cpu().tolist())
                
                all_val_preds = torch.tensor(all_val_preds)
                all_val_labels = torch.tensor(all_val_labels)
                
                print(f"  [Val Debug] Samples: {len(all_val_labels)}, "
                      f"Labels: {torch.bincount(all_val_labels, minlength=len(cfg.labels)).tolist()}, "
                      f"Preds: {torch.bincount(all_val_preds, minlength=len(cfg.labels)).tolist()}, "
                      f"Correct: {(all_val_preds == all_val_labels).sum().item()}")
        
        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        
        # Save if validation loss improved (more reliable than accuracy for binary classification)
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            improved = True
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': cfg.model.model_type,
                'use_2d_conv': cfg.model.use_2d_conv,
                'num_classes': cfg.model.num_classes,
                'config': cfg,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})\n")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping: No improvement for {early_stop_patience} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.3f}\n")
                break
            print(f"  (No improvement, patience: {patience_counter}/{early_stop_patience})\n")
    
    print(f"\n{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    
    # Load best model and test
    # weights_only=False is safe here since it's our own checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f} acc={test_acc:.3f}")
    print(f"{'='*60}\n")
    
    # Calculate detailed metrics and plots
    print("Calculating detailed metrics...")
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        config=cfg,
        save_plots=True,
        plot_dir=str(plot_dir)
    )
    
    print(f"\n{'='*60}")
    print("Metrics Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    if len(cfg.labels) > 2:
        print(f"\nPer-class metrics:")
        for i, label in enumerate(cfg.labels):
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
            print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
            print(f"    F1-Score: {metrics['f1_per_class'][i]:.4f}")
            print(f"    Support: {metrics['support_per_class'][i]}")
    
    print(f"\nPlots saved to: {plot_dir}")
    print(f"{'='*60}\n")
    
    # Export models
    print("Exporting models...")
    example_features, _ = next(iter(test_loader))
    if example_features.dim() == 2:
        example_features = example_features.unsqueeze(1)
    elif example_features.dim() == 3 and example_features.shape[1] != 1:
        example_features = example_features.unsqueeze(1)
    
    example_input = example_features[:1].to("cpu")
    export_paths = export_models(model, cfg, example_input, device)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_keyword_spotter()


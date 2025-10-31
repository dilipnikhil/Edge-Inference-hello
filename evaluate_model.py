"""
Standalone script to evaluate a trained model and generate metrics plots
"""

import torch
import argparse
from pathlib import Path

from config import ProjectConfig
from model_2d import build_model
from model import TinyKWS, MinimalKWS
from data_loader import create_dataloaders
from metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model and generate metrics")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Root directory for Speech Commands dataset")
    parser.add_argument("--hello_dir", type=str, default=None,
                       help="Directory with hello audio files")
    parser.add_argument("--other_dir", type=str, default=None,
                       help="Directory with other audio files")
    parser.add_argument("--use_speech_commands", action="store_true",
                       help="Use Speech Commands dataset")
    parser.add_argument("--plot_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test",
                       help="Which split to evaluate on")
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    
    # Get config
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        cfg = ProjectConfig()
        # Try to infer from checkpoint
        num_classes = checkpoint.get("num_classes", 2)
        cfg.model.num_classes = num_classes
        if num_classes == 2:
            cfg.labels = ("other", "hello")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        project_cfg=cfg,
        batch_size=32,
        root=args.data_root,
        hello_dir=args.hello_dir,
        other_dir=args.other_dir,
        use_speech_commands=args.use_speech_commands or cfg.training.use_speech_commands,
    )
    
    # Select appropriate loader
    if args.split == "train":
        eval_loader = train_loader
    elif args.split == "val":
        eval_loader = val_loader
    else:
        eval_loader = test_loader
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    use_2d = checkpoint.get("use_2d_conv", False)
    model_type = checkpoint.get("model_type", "tiny").replace("_quantized", "")
    num_classes = checkpoint.get("num_classes", cfg.model.num_classes)
    
    if use_2d:
        cfg.model.use_2d_conv = True
        cfg.model.model_type = model_type
        cfg.model.num_classes = num_classes
        model = build_model(cfg.model)
    else:
        if model_type == "tiny":
            model = TinyKWS(num_classes=num_classes, num_filters=64)
        else:
            model = MinimalKWS(num_classes=num_classes, num_filters=32)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model: {model_type}, 2D: {use_2d}, Classes: {num_classes}")
    print(f"Evaluating on {args.split} set ({len(eval_loader.dataset)} samples)\n")
    
    # Evaluate
    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = evaluate_model(
        model,
        eval_loader,
        device,
        config=cfg,
        save_plots=True,
        plot_dir=args.plot_dir
    )
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Metrics plots saved to: {args.plot_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Train ASR models with DREAM.

Supports:
- Standard DREAM Stack
- Coordinated DREAM Stack (with top-down modulation)
- Experiment flags (disable fast weights, LTC, sleep, etc.)

Usage:
------
# Full dataset, standard DREAM
python train.py --root /content/drive/MyDrive/dream/dataset/ljspeech \
                --log-dir /content/drive/MyDrive/dream/experiments/dream_base

# Subset mode (quick experiment)
python train.py --root /content/drive/MyDrive/dream/dataset/ljspeech \
                --subset 100 \
                --log-dir /content/drive/MyDrive/dream/experiments/dream_subset

# Coordinated model
python train.py --root /content/drive/MyDrive/dream/dataset/ljspeech \
                --model coordinated \
                --log-dir /content/drive/MyDrive/dream/experiments/dream_coordinated

# Disable fast weights (ablation study)
python train.py --root /content/drive/MyDrive/dream/dataset/ljspeech \
                --no-fast-weights \
                --log-dir /content/drive/MyDrive/dream/experiments/dream_no_fw

# Run both models in parallel
python train.py --root ... --model dream --run-name dream_base &
python train.py --root ... --model coordinated --run-name dream_coord &
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.dataset import create_dataloaders
from examples.model import create_model
from examples.trainer import ASRTrainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASR with DREAM")
    
    # =====================================================================
    # Data parameters
    # =====================================================================
    parser.add_argument("--root", type=str, required=True,
                        help="Path to LJSpeech directory (contains wavs/ and metadata.csv)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit dataset to N samples (for quick experiments)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split (default: 0.1)")
    
    # =====================================================================
    # Model parameters
    # =====================================================================
    parser.add_argument("--model", type=str, default="dream",
                        choices=["dream", "coordinated"],
                        help="Model type (default: dream)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256],
                        help="Hidden dimensions for each layer (default: [256, 256, 256])")
    parser.add_argument("--rank", type=int, default=16,
                        help="Fast weights rank (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    
    # Block control flags
    parser.add_argument("--no-fast-weights", action="store_true",
                        help="Disable fast weights (ablation study)")
    parser.add_argument("--no-ltc", action="store_true",
                        help="Disable Liquid Time-Constants")
    parser.add_argument("--no-sleep", action="store_true",
                        help="Disable sleep consolidation")
    parser.add_argument("--freeze-fast-weights", action="store_true",
                        help="Freeze fast weights during training (static base)")
    
    # Coordinated model flags
    parser.add_argument("--no-hierarchical-tau", action="store_true",
                        help="Disable hierarchical tau (coordinated only)")
    parser.add_argument("--no-inter-layer-prediction", action="store_true",
                        help="Disable inter-layer prediction loss (coordinated only)")
    
    # =====================================================================
    # Training parameters
    # =====================================================================
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (default: 1e-5)")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Gradient clipping (default: 5.0)")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision")
    
    # =====================================================================
    # Logging and checkpoints
    # =====================================================================
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (default: auto-generated)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches (default: 10)")
    parser.add_argument("--example-interval", type=int, default=5,
                        help="Show prediction examples every N epochs (default: 5)")
    
    # =====================================================================
    # Reproducibility
    # =====================================================================
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # =====================================================================
    # Debug
    # =====================================================================
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (extra logging)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    set_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"DREAM ASR Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_str = args.model
        if args.no_fast_weights:
            model_str += "_no_fw"
        if args.no_ltc:
            model_str += "_no_ltc"
        if args.no_sleep:
            model_str += "_no_sleep"
        if args.freeze_fast_weights:
            model_str += "_freeze_fw"
        if args.subset:
            model_str += f"_subset{args.subset}"
        args.run_name = f"{model_str}_{timestamp}"
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset,
        val_split=args.val_split
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(
        model_type=args.model,
        n_mels=80,
        hidden_dims=args.hidden_dims,
        rank=args.rank,
        num_classes=27,  # 26 letters + space
        dropout=args.dropout,
        use_fast_weights=not args.no_fast_weights,
        use_ltc=not args.no_ltc,
        use_sleep=not args.no_sleep,
        use_hierarchical_tau=not args.no_hierarchical_tau,
        use_inter_layer_prediction=not args.no_inter_layer_prediction,
        freeze_fast_weights=args.freeze_fast_weights
    )
    
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"\nConfiguration:")
    print(f"  Fast weights: {not args.no_fast_weights}")
    print(f"  LTC: {not args.no_ltc}")
    print(f"  Sleep: {not args.no_sleep}")
    if args.model == 'coordinated':
        print(f"  Hierarchical tau: {not args.no_hierarchical_tau}")
        print(f"  Inter-layer prediction: {not args.no_inter_layer_prediction}")
    if args.freeze_fast_weights:
        print(f"  Fast weights frozen: True")
    
    # Create trainer
    trainer = ASRTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
        log_dir=args.log_dir,
        device=device
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        run_name=args.run_name,
        example_interval=args.example_interval
    )
    
    # Save final model
    final_path = os.path.join(args.log_dir, args.run_name, 'final.pt')
    trainer.save_checkpoint(os.path.join(args.log_dir, args.run_name), 'final')
    print(f"\nSaved final model: {final_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Run: {args.run_name}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

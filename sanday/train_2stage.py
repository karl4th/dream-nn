"""
2-Stage Training for SANDAY Phoneme Recognition.

Stage 1: Pre-training
  - Fast weights: FROZEN
  - Loss: MSE reconstruction
  - LR: 0.001
  - Epochs: 20-50
  - Goal: Learn slow weights (C, W, B)

Stage 2: Adaptation Training
  - Fast weights: ACTIVE
  - Loss: CTC
  - LR: 0.0001
  - Epochs: 50-100
  - Goal: Learn fast weights plasticity + phoneme recognition
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SandayASR
from .phonemes import EnglishPhonemes
from .data import LJSpeechDataset, create_dataloader


class TwoStageTrainer:
    """
    2-Stage Training for SANDAY.
    """
    
    def __init__(self,
                 model: SandayASR,
                 phonemes: EnglishPhonemes,
                 device: str = 'cuda',
                 ):
        self.model = model.to(device)
        self.phonemes = phonemes
        self.device = device
        
        self.history = {
            'stage1_loss': [],
            'stage2_loss': [],
            'stage2_val_loss': [],
        }
    
    def train_stage1(self,
                    train_loader: DataLoader,
                    n_epochs: int = 20,
                    lr: float = 0.001,
                    save_dir: Optional[str] = None,
                    ) -> Dict:
        """
        Stage 1: Pre-training with reconstruction loss.
        
        Fast weights are FROZEN. Only slow weights (C, W, B) are trained.
        """
        print(f"\n{'='*70}")
        print(f"STAGE 1: PRE-TRAINING ({n_epochs} epochs)")
        print(f"{'='*70}")
        print(f"  - Fast weights: FROZEN")
        print(f"  - Loss: MSE Reconstruction")
        print(f"  - LR: {lr}")
        print(f"{'='*70}\n")
        
        # Switch to pre-training mode
        self.model.switch_to_pretraining(use_mse=True)
        
        # Optimizer: ONLY slow weights (dream parameters)
        optimizer = torch.optim.Adam(
            list(self.model.dream_stack.parameters()) +
            list(self.model.reconstruction_head.parameters()),
            lr=lr,
        )
        
        self.model.train()
        
        for epoch in range(n_epochs):
            start_time = time.time()
            total_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward with stage='pretrain'
                outputs = self.model(features, stage='pretrain')
                
                loss = outputs['recon_loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            self.history['stage1_loss'].append(avg_loss)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.4f} ({elapsed:.1f}s)")
        
        print(f"\n✅ Stage 1 complete! Best loss: {min(self.history['stage1_loss']):.4f}")
        
        # Save checkpoint
        if save_dir:
            save_path = Path(save_dir) / 'stage1_pretrain.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': self.history['stage1_loss'],
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")
        
        return self.history
    
    def train_stage2(self,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    n_epochs: int = 50,
                    lr: float = 0.0001,
                    save_dir: Optional[str] = None,
                    ) -> Dict:
        """
        Stage 2: Adaptation training with CTC loss.
        
        Fast weights are ACTIVE. All parameters trained.
        """
        print(f"\n{'='*70}")
        print(f"STAGE 2: ADAPTATION TRAINING ({n_epochs} epochs)")
        print(f"{'='*70}")
        print(f"  - Fast weights: ACTIVE (plasticity enabled)")
        print(f"  - Loss: CTC")
        print(f"  - LR: {lr}")
        print(f"{'='*70}\n")
        
        # Switch to adaptation mode
        self.model.switch_to_adaptation()
        
        # Optimizer: ALL parameters (including fast weights plasticity)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        
        # CTC loss
        ctc_loss = nn.CTCLoss(blank=1, zero_infinity=True)
        
        best_val_loss = float('inf')
        
        self.model.train()
        
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                feat_lengths = batch['feature_lengths'].to(self.device)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward with stage='adaptation'
                outputs = self.model(features, stage='adaptation')
                
                # CTC loss
                log_probs = outputs['log_probs'].transpose(0, 1)  # (time, batch, num_phonemes)
                loss = ctc_loss(
                    log_probs.log(),
                    phonemes,
                    feat_lengths,
                    phoneme_lengths,
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches
            
            # Validate
            val_loss = self.validate(val_loader, ctc_loss)
            self.history['stage2_loss'].append(avg_train_loss)
            self.history['stage2_val_loss'].append(val_loss)
            
            elapsed = time.time() - start_time
            
            # Save best
            if val_loss < best_val_loss and save_dir:
                best_val_loss = val_loss
                save_path = Path(save_dir) / 'stage2_adaptation_best.pt'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"  ✅ val_loss improved to {val_loss:.4f}")
            
            print(f"Epoch {epoch+1:3d}/{n_epochs}: train={avg_train_loss:.4f}, val={val_loss:.4f} ({elapsed:.1f}s)")
        
        print(f"\n✅ Stage 2 complete! Best val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if save_dir:
            save_path = Path(save_dir) / 'stage2_adaptation_last.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")
        
        return self.history
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, ctc_loss) -> float:
        """Validate Stage 2."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_loader:
            features = batch['features'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            feat_lengths = batch['feature_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            outputs = self.model(features, stage='adaptation')
            
            log_probs = outputs['log_probs'].transpose(0, 1)
            loss = ctc_loss(
                log_probs.log(),
                phonemes,
                feat_lengths,
                phoneme_lengths,
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        self.model.train()
        return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='2-Stage SANDAY Training')
    
    # Data
    parser.add_argument('--audio-dir', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None,
                       help='Use only N samples (for testing)')
    parser.add_argument('--max-audio-len', type=float, default=5.0)
    
    # Model
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--rank', type=int, default=16)
    
    # Stage 1
    parser.add_argument('--stage1-epochs', type=int, default=20)
    parser.add_argument('--stage1-lr', type=float, default=0.001)
    
    # Stage 2
    parser.add_argument('--stage2-epochs', type=int, default=50)
    parser.add_argument('--stage2-lr', type=float, default=0.0001)
    
    # Common
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='sanday/checkpoints_2stage')
    
    args = parser.parse_args()
    
    # Device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Metadata path
    if args.metadata_path is None:
        args.metadata_path = str(Path(args.audio_dir).parent / 'metadata.csv')
    
    # Phonemes
    phonemes = EnglishPhonemes(use_stress=False)
    print(f"Phoneme vocabulary: {len(phonemes)} classes")
    
    # Dataset
    print(f"\nLoading dataset...")
    dataset = LJSpeechDataset(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
        phonemes=phonemes,
        max_audio_len=args.max_audio_len,
    )
    
    # Subset
    if args.subset is not None:
        n_samples = min(args.subset, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(n_samples))
        print(f"Using subset: {n_samples} samples")
    
    # Split
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Val: {n_val}")
    
    # DataLoaders
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Model
    model = SandayASR(
        input_dim=80,
        hidden_dims=args.hidden_dims,
        rank=args.rank,
        num_phonemes=len(phonemes),
        freeze_fast_weights=True,  # Start with frozen (Stage 1)
    )
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Trainer
    trainer = TwoStageTrainer(model, phonemes, device=device)
    
    # Save dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # STAGE 1
    trainer.train_stage1(
        train_loader,
        n_epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        save_dir=str(save_dir),
    )
    
    # STAGE 2
    trainer.train_stage2(
        train_loader,
        val_loader,
        n_epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        save_dir=str(save_dir),
    )
    
    # Save history
    history_path = save_dir / 'training_history_2stage.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"\n📊 History saved to: {history_path}")
    
    print(f"\n{'='*70}")
    print(f"2-STAGE TRAINING COMPLETE!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

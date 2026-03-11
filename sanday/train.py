"""
Training script for SANDAY phoneme recognition.

Usage:
    uv run python -m sanday.train --audio-dir /path/to/LJSpeech-1.1/wavs
    OR
    cd sanday && python train.py --audio-dir /path/to/LJSpeech-1.1/wavs
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


class CTCWithAlignment(nn.Module):
    """
    CTC loss with optional alignment tracking.
    """
    
    def __init__(self, blank_id: int = 1):
        super().__init__()
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    
    def forward(self,
                log_probs: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor,
                ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities (batch, time, num_phonemes)
        targets : torch.Tensor
            Target phoneme IDs (batch, target_len)
        input_lengths : torch.Tensor
            Lengths of input sequences
        target_lengths : torch.Tensor
            Lengths of target sequences
        
        Returns
        -------
        loss : torch.Tensor
            CTC loss
        """
        # Transpose to (time, batch, num_phonemes)
        log_probs = log_probs.transpose(0, 1)
        
        # Compute loss
        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        
        return loss


class SandayTrainer:
    """
    Trainer for SANDAY phoneme recognition model.
    """
    
    def __init__(self,
                 model: SandayASR,
                 phonemes: EnglishPhonemes,
                 device: str = 'cuda',
                 lr: float = 0.001,
                 grad_clip: float = 5.0,
                 ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : SandayASR
            The model to train
        phonemes : EnglishPhonemes
            Phoneme converter
        device : str
            Device to use
        lr : float
            Learning rate
        grad_clip : float
            Gradient clipping value
        """
        self.model = model.to(device)
        self.phonemes = phonemes
        self.device = device
        self.lr = lr
        self.grad_clip = grad_clip
        
        # Loss and optimizer
        self.criterion = CTCWithAlignment(blank_id=1)  # blank_id = 1
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = 5
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                ),
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                ),
            ],
            milestones=[self.warmup_epochs],
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'ctc_stats': [],
            'dream_stats': [],
        }
    
    def compute_ctc_stats(self, log_probs: torch.Tensor) -> Dict:
        """
        Compute CTC output statistics.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities (batch, time, num_phonemes)
        
        Returns
        -------
        stats : Dict
            CTC statistics
        """
        blank_id = 1
        batch_size, time_steps, num_phonemes = log_probs.shape
        
        # Get predictions
        pred_ids = log_probs.argmax(dim=-1)  # (batch, time)
        
        # Count blanks
        blank_count = (pred_ids == blank_id).sum().item()
        total_elements = pred_ids.numel()
        blank_ratio = blank_count / total_elements
        
        # Count repeats (consecutive same predictions)
        repeat_count = 0
        for b in range(batch_size):
            for t in range(1, time_steps):
                if pred_ids[b, t] == pred_ids[b, t-1] and pred_ids[b, t] != blank_id:
                    repeat_count += 1
        
        non_blank = total_elements - blank_count
        repeat_ratio = repeat_count / max(1, non_blank)
        
        # Vocabulary usage
        unique_preds = pred_ids.unique()
        non_blank_preds = unique_preds[unique_preds != blank_id]
        vocab_usage = len(non_blank_preds) / (num_phonemes - 1) * 100
        
        return {
            'blank_ratio': blank_ratio,
            'repeat_ratio': repeat_ratio * 100,
            'vocab_usage': vocab_usage,
            'unique_phonemes': len(non_blank_preds),
            'total_phonemes': num_phonemes - 1,
        }
    
    def compute_dream_stats(self, states, batch_size: int) -> Dict:
        """
        Compute DREAM layer statistics (tau, surprise, error).
        
        Parameters
        ----------
        states : CoordinatedState
            DREAM states
        batch_size : int
            Batch size
        
        Returns
        -------
        stats : Dict
            Per-layer statistics
        """
        layer_stats = []
        
        for i, layer_state in enumerate(states.layer_states):
            # Get cell for tau parameters
            cell = self.model.dream_stack.cells[i]
            
            # Effective tau (average across batch and time)
            tau_base = cell.tau_sys.item() * cell.tau_depth_factor
            surprise = layer_state.avg_surprise.mean().item()
            tau_surprise_scale = cell.tau_surprise_scale.item()
            tau_eff = tau_base / (1.0 + surprise * tau_surprise_scale)
            
            # Error norm
            error_norm = layer_state.error_mean.norm(dim=-1).mean().item()
            
            layer_stats.append({
                'layer': i,
                'tau_eff': tau_eff,
                'tau_base': tau_base,
                'surprise': surprise,
                'error': error_norm,
            })
        
        return {'layers': layer_stats}
    
    @torch.no_grad()
    def log_epoch_stats(self, epoch: int, val_loader: DataLoader):
        """
        Log detailed statistics after each epoch.
        """
        self.model.eval()
        
        # Get a batch for statistics
        batch = next(iter(val_loader))
        features = batch['features'].to(self.device)
        phonemes = batch['phonemes'].to(self.device)
        feat_lengths = batch['feature_lengths'].to(self.device)
        phoneme_lengths = batch['phoneme_lengths'].to(self.device)
        
        # Forward pass with states
        outputs = self.model(features, lengths=feat_lengths, return_states=True)
        
        # CTC statistics
        ctc_stats = self.compute_ctc_stats(outputs['log_probs'])
        self.history['ctc_stats'].append(ctc_stats)
        
        # DREAM statistics
        dream_stats = self.compute_dream_stats(outputs['states'], features.shape[0])
        self.history['dream_stats'].append(dream_stats)
        
        # Print logs
        print("\n" + "·" * 70)
        print(f"📈 CTC STATS (Epoch {epoch})")
        print(f"   Blank Ratio:  {ctc_stats['blank_ratio']:.4f}")
        print(f"   Repeat Ratio: {ctc_stats['repeat_ratio']:.2f}x")
        print(f"   Vocab Usage:  {ctc_stats['vocab_usage']:.1f}% ({ctc_stats['unique_phonemes']}/{ctc_stats['total_phonemes']})")
        
        print(f"\n🧠 DREAM DYNAMICS (Effective Tau & Surprise)")
        for layer_stat in dream_stats['layers']:
            tau_ms = layer_stat['tau_eff'] * 10  # Approximate ms
            print(f"   Layer {layer_stat['layer']}: τ={layer_stat['tau_eff']:.3f} ({tau_ms:>6.1f}ms) | "
                  f"Surprise={layer_stat['surprise']:.4f} | Error={layer_stat['error']:.2f}")
        
        # Prediction examples
        print(f"\n📝 PREDICTION EXAMPLES")
        self.model.eval()
        
        for idx in range(min(3, features.shape[0])):
            feat = features[idx:idx+1]
            target_ids = phonemes[idx, :phoneme_lengths[idx]]
            
            # Decode
            pred_phonemes, _ = self.model.decode(feat, self.phonemes)
            true_phonemes = self.phonemes.ids_to_phonemes(target_ids.cpu().tolist())
            
            # Truncate long sequences for display
            max_display = 80
            true_str = ' '.join(true_phonemes[:max_display])
            if len(true_phonemes) > max_display:
                true_str += '...'
            
            pred_str = ' '.join(pred_phonemes[:max_display])
            if len(pred_phonemes) > max_display:
                pred_str += '...'
            
            print(f"Sample {idx+1}:")
            print(f"  Target:    {true_str}")
            print(f"  Predicted: {pred_str}")
        
        print("······································································\n")
    
    def train_epoch(self, loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        avg_loss : float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            features = batch['features'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            feat_lengths = batch['feature_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features, lengths=feat_lengths)
            
            # Compute loss
            loss = self.criterion(
                outputs['log_probs'],
                phonemes,
                feat_lengths,
                phoneme_lengths,
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.history['train_loss'].append(avg_loss)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """
        Validate on validation set.
        
        Returns
        -------
        avg_loss : float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            features = batch['features'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            feat_lengths = batch['feature_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward pass
            outputs = self.model(features, lengths=feat_lengths)
            
            # Compute loss
            loss = self.criterion(
                outputs['log_probs'],
                phonemes,
                feat_lengths,
                phoneme_lengths,
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              n_epochs: int = 50,
              save_dir: Optional[str] = None,
              ) -> Dict:
        """
        Full training loop.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        n_epochs : int
            Number of epochs
        save_dir : str, optional
            Directory to save checkpoints
        
        Returns
        -------
        history : Dict
            Training history
        """
        print(f"\n{'='*70}")
        print(f"SANDAY Training - {n_epochs} epochs")
        print(f"{'='*70}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate and log
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.scheduler.step(val_loss)
                
                # Log detailed stats
                self.log_epoch_stats(epoch + 1, val_loader)
                
                # Save best model
                if val_loss < best_val_loss and save_dir:
                    best_val_loss = val_loss
                    save_path = Path(save_dir) / 'checkpoint_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
                    print(f"  ✅ val_loss improved to {val_loss:.6f}")
                    print(f"  💾 Checkpoint saved: {save_path}")
                
                # Save last checkpoint
                if save_dir:
                    save_path = Path(save_dir) / 'checkpoint_last.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
                    print(f"  💾 Checkpoint saved: {save_path}")
            else:
                self.scheduler.step(train_loss)
                val_loss = None
            
            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{n_epochs}")
            print(f"   Train Loss: {train_loss:.4f} ({elapsed:.1f}s)")
            if val_loss is not None:
                print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   LR:         {current_lr:.2e}")
        
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        return self.history


def main():
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('cmudict')
    parser = argparse.ArgumentParser(description='Train SANDAY phoneme recognizer')
    
    # Data arguments
    parser.add_argument('--audio-dir', type=str, required=True,
                       help='Path to LJSpeech wav files')
    parser.add_argument('--metadata-path', type=str, default=None,
                       help='Path to metadata.csv (default: audio_dir/../metadata.csv)')
    parser.add_argument('--max-audio-len', type=float, default=5.0,
                       help='Maximum audio length in seconds')
    
    # Model arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256, 256],
                       help='Hidden dimensions for DREAM layers')
    parser.add_argument('--rank', type=int, default=16,
                       help='Fast weights rank')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use only N samples from dataset (for testing)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,  # Changed from 0.001 to 0.0001
                       help='Learning rate (start with 1e-4 for CTC stability)')
    parser.add_argument('--grad-clip', type=float, default=1.0,  # Changed from 5.0 to 1.0
                       help='Gradient clipping (lower for stability)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Epochs with reduced learning rate (warmup)')
    parser.add_argument('--num-workers', type=int, default=0,  # Changed from 4 to 0 (no workers = more stable)
                       help='Number of data loading workers (0 = main process only)')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='sanday/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set metadata path
    if args.metadata_path is None:
        args.metadata_path = str(Path(args.audio_dir).parent / 'metadata.csv')
    
    # Create phoneme converter
    phonemes = EnglishPhonemes(use_stress=False)
    print(f"Phoneme vocabulary: {len(phonemes)} classes")
    
    # Create dataset
    print(f"\nLoading dataset from {args.audio_dir}...")
    dataset = LJSpeechDataset(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
        phonemes=phonemes,
        max_audio_len=args.max_audio_len,
    )
    
    # Subset (for testing)
    if args.subset is not None:
        n_samples = min(args.subset, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(n_samples))
        print(f"Using subset: {n_samples}/{len(dataset)} samples")
    
    # Split train/val
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {n_train} samples, Val: {n_val} samples")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Create model (coordination ALWAYS enabled - it's our key feature!)
    model = SandayASR(
        input_dim=80,
        hidden_dims=args.hidden_dims,
        rank=args.rank,
        num_phonemes=len(phonemes),
        dropout=args.dropout,
        use_coordination=True,  # ALWAYS ON!
    )
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Coordination: ✅ ENABLED (hierarchical tau + inter-layer prediction)")
    
    # Create trainer
    trainer = SandayTrainer(
        model,
        phonemes,
        device=device,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    
    # Train
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        save_dir=str(save_dir),
    )
    
    # Save history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Test inference
    print("\n" + "="*70)
    print("Testing inference...")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        # Get a sample from validation set
        sample = val_dataset[0]
        features, phoneme_ids, audio_id = sample
        features = features.unsqueeze(0).to(device)
        
        # Decode
        pred_phonemes, aligned = model.decode(features, phonemes)
        true_phonemes = phonemes.ids_to_phonemes(phoneme_ids.tolist())
        
        print(f"\nAudio ID: {audio_id}")
        print(f"Predicted: {' '.join(pred_phonemes)}")
        print(f"True:      {' '.join(true_phonemes)}")


if __name__ == '__main__':
    main()

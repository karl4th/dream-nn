"""
ASR Trainer with CTC Loss.

Supports:
- DREAM and Coordinated DREAM models
- Detailed logging for debugging
- Checkpointing
- Mixed precision training (optional)
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class ASRTrainer:
    """
    Trainer for ASR with CTC loss.
    
    Parameters
    ----------
    model : nn.Module
        ASR model (DREAMASR or CoordinatedDREAMASR)
    learning_rate : float
        Learning rate
    weight_decay : float
        Weight decay for optimizer
    grad_clip : float
        Gradient clipping value
    use_amp : bool
        Use automatic mixed precision
    log_dir : str
        Directory for logs and checkpoints
    device : str
        Device to use ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        use_amp: bool = False,
        log_dir: str = "./logs",
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.log_dir = log_dir
        self.grad_clip = grad_clip
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.log_data = {
            'train_loss': [],
            'val_loss': [],
            'train_ctc_loss': [],
            'val_ctc_loss': [],
            'train_aux_loss': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns
        -------
        metrics : dict
            Average losses and metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (mel_spec, tokens, mel_lengths, token_lengths) in enumerate(train_loader):
            mel_spec = mel_spec.to(self.device)
            tokens = tokens.to(self.device)
            mel_lengths = mel_lengths.to(self.device)
            token_lengths = token_lengths.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    if hasattr(self.model, 'forward') and 'coordinated' in str(type(self.model)).lower():
                        log_probs, out_lengths, aux_losses = self.model(
                            mel_spec, mel_lengths, return_losses=True
                        )
                    else:
                        log_probs, out_lengths = self.model(mel_spec, mel_lengths)
                        aux_losses = None
                    
                    # CTC loss
                    ctc_loss = self.ctc_loss(
                        log_probs.transpose(0, 1),  # (time, batch, classes)
                        tokens,
                        out_lengths,
                        token_lengths
                    )
                    
                    # Add auxiliary losses for coordinated model
                    total_batch_loss = ctc_loss
                    aux_loss_val = 0.0
                    
                    if aux_losses is not None:
                        # Weight auxiliary losses
                        aux_loss_val = aux_losses.get('inter_layer', 0.0) * 0.01
                        aux_loss_val += aux_losses.get('reconstruction', 0.0) * 0.1
                        total_batch_loss = total_batch_loss + aux_loss_val
            
            else:
                if hasattr(self.model, 'forward') and 'coordinated' in str(type(self.model)).lower():
                    log_probs, out_lengths, aux_losses = self.model(
                        mel_spec, mel_lengths, return_losses=True
                    )
                else:
                    log_probs, out_lengths = self.model(mel_spec, mel_lengths)
                    aux_losses = None
                
                # CTC loss
                ctc_loss = self.ctc_loss(
                    log_probs.transpose(0, 1),
                    tokens,
                    out_lengths,
                    token_lengths
                )
                
                # Add auxiliary losses
                total_batch_loss = ctc_loss
                aux_loss_val = 0.0
                
                if aux_losses is not None:
                    aux_loss_val = aux_losses.get('inter_layer', 0.0) * 0.01
                    aux_loss_val += aux_losses.get('reconstruction', 0.0) * 0.1
                    total_batch_loss = total_batch_loss + aux_loss_val
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += total_batch_loss.item()
            total_ctc_loss += ctc_loss.item()
            total_aux_loss += aux_loss_val
            num_batches += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_batch_loss.item():.4f} | "
                      f"CTC: {ctc_loss.item():.4f} | "
                      f"Aux: {aux_loss_val:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        epoch_time = time.time() - start_time
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches,
            'epoch_time': epoch_time
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        return_examples: bool = False,
        num_examples: int = 5
    ) -> Dict:
        """
        Validate on validation set.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        return_examples : bool
            Return prediction examples (text)
        num_examples : int
            Number of examples to return
        
        Returns
        -------
        metrics : dict
            Validation metrics
        examples : list, optional
            List of (true_text, predicted_text) tuples
        """
        self.model.eval()
        
        total_loss = 0.0
        total_ctc_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        examples = []
        
        for batch_idx, (mel_spec, tokens, mel_lengths, token_lengths) in enumerate(val_loader):
            mel_spec = mel_spec.to(self.device)
            tokens = tokens.to(self.device)
            mel_lengths = mel_lengths.to(self.device)
            token_lengths = token_lengths.to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'coordinated' in str(type(self.model)).lower():
                log_probs, out_lengths, aux_losses = self.model(
                    mel_spec, mel_lengths, return_losses=True
                )
            else:
                log_probs, out_lengths = self.model(mel_spec, mel_lengths)
                aux_losses = None
            
            # CTC loss
            ctc_loss = self.ctc_loss(
                log_probs.transpose(0, 1),
                tokens,
                out_lengths,
                token_lengths
            )
            
            # Add auxiliary losses
            total_batch_loss = ctc_loss
            aux_loss_val = 0.0
            
            if aux_losses is not None:
                aux_loss_val = aux_losses.get('inter_layer', 0.0) * 0.01
                aux_loss_val += aux_losses.get('reconstruction', 0.0) * 0.1
                total_batch_loss = total_batch_loss + aux_loss_val
            
            # Accumulate
            total_loss += total_batch_loss.item()
            total_ctc_loss += ctc_loss.item()
            total_aux_loss += aux_loss_val
            num_batches += 1
            
            # Decode examples
            if return_examples and len(examples) < num_examples:
                batch_examples = self._decode_batch(log_probs, tokens, out_lengths, token_lengths)
                examples.extend(batch_examples[:num_examples - len(examples)])
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches
        }
        
        if return_examples:
            return metrics, examples[:num_examples]
        
        return metrics
    
    def _decode_batch(
        self,
        log_probs: torch.Tensor,
        true_tokens: torch.Tensor,
        out_lengths: torch.Tensor,
        token_lengths: torch.Tensor,
        blank_id: int = 0
    ) -> list:
        """
        Decode CTC predictions to text.
        
        Parameters
        ----------
        log_probs : torch.Tensor
            Log probabilities (batch, time, num_classes)
        true_tokens : torch.Tensor
            True token IDs (batch, text_length)
        out_lengths : torch.Tensor
            Output lengths
        token_lengths : torch.Tensor
            True token lengths
        blank_id : int
            CTC blank token ID
        
        Returns
        -------
        examples : list
            List of (true_text, predicted_text, confidence) tuples
        """
        # Get vocabulary from model or dataset
        if hasattr(self.model, 'num_classes'):
            num_classes = self.model.num_classes
        else:
            num_classes = 27
        
        # Simple character vocabulary
        chars = " abcdefghijklmnopqrstuvwxyz"
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        examples = []
        batch_size = log_probs.shape[0]
        
        for i in range(min(batch_size, 5)):
            # Get predictions for this sample
            probs = log_probs[i, :out_lengths[i], :]  # (time, num_classes)
            
            # Greedy decoding: take argmax at each timestep
            pred_ids = probs.argmax(dim=-1).cpu().tolist()
            
            # CTC decoding: remove blanks and repeated characters
            pred_text = self._ctc_decode(pred_ids, idx_to_char, blank_id)
            
            # Get true text
            true_ids = true_tokens[i, :token_lengths[i]].cpu().tolist()
            true_text = ''.join(idx_to_char.get(id, '?') for id in true_ids)
            
            # Compute confidence (average probability of predicted tokens)
            confidences = probs.max(dim=-1).values
            confidence = confidences.mean().item()
            
            examples.append((true_text, pred_text, confidence))
        
        return examples
    
    def _ctc_decode(self, pred_ids: list, idx_to_char: dict, blank_id: int = 0) -> str:
        """
        Decode CTC output (remove blanks and repeated characters).
        
        Parameters
        ----------
        pred_ids : list
            Predicted token IDs
        idx_to_char : dict
            ID to character mapping
        blank_id : int
            CTC blank token ID
        
        Returns
        -------
        text : str
            Decoded text
        """
        result = []
        prev_id = -1
        
        for id_ in pred_ids:
            # Skip blanks and repeated characters
            if id_ != prev_id and id_ != blank_id:
                result.append(idx_to_char.get(id_, '?'))
            prev_id = id_
        
        return ''.join(result)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        run_name: Optional[str] = None,
        example_interval: int = 5
    ):
        """
        Full training loop.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Number of epochs to train
        run_name : str, optional
            Name for this run (for logging)
        example_interval : int
            Show prediction examples every N epochs (default: 5)
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_name = run_name
        run_dir = os.path.join(self.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training Run: {run_name}")
        print(f"{'='*60}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Example interval: every {example_interval} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            show_examples = ((epoch + 1) % example_interval == 0) or (epoch + 1 == num_epochs)
            
            if show_examples:
                val_metrics, examples = self.validate(val_loader, return_examples=True)
            else:
                val_metrics = self.validate(val_loader, return_examples=False)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.log_data['train_loss'].append(train_metrics['loss'])
            self.log_data['val_loss'].append(val_metrics['loss'])
            self.log_data['train_ctc_loss'].append(train_metrics['ctc_loss'])
            self.log_data['val_ctc_loss'].append(val_metrics['ctc_loss'])
            self.log_data['train_aux_loss'].append(train_metrics['aux_loss'])
            self.log_data['learning_rate'].append(current_lr)
            self.log_data['epoch_time'].append(train_metrics['epoch_time'])
            
            # Print summary
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
                  f"CTC: {train_metrics['ctc_loss']:.4f} | "
                  f"Aux: {train_metrics['aux_loss']:.4f} | "
                  f"Time: {train_metrics['epoch_time']:.1f}s")
            print(f"Val Loss:   {val_metrics['loss']:.4f} | "
                  f"CTC: {val_metrics['ctc_loss']:.4f} | "
                  f"Aux: {val_metrics['aux_loss']:.4f}")
            print(f"LR: {current_lr:.2e}")
            
            # Print examples
            if show_examples and examples:
                print(f"\n{'='*60}")
                print(f"Prediction Examples (Epoch {epoch+1}):")
                print(f"{'='*60}")
                
                for i, (true_text, pred_text, confidence) in enumerate(examples, 1):
                    # Clean up text for display
                    true_display = true_text.replace('\n', '\\n')[:60]
                    pred_display = pred_text.replace('\n', '\\n')[:60]
                    
                    # Color coding based on match
                    match = "✓" if true_text.strip() == pred_text.strip() else "✗"
                    
                    print(f"\n  {match} Example {i} (conf: {confidence:.2f}):")
                    print(f"    True:  '{true_display}'")
                    print(f"    Pred:  '{pred_display}'")
                
                print(f"\n{'='*60}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                self.save_checkpoint(run_dir, 'best', is_best=True)
                print(f"  ★ New best model! Val loss: {val_metrics['loss']:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(run_dir, f'epoch_{epoch+1}')
            
            # Save metrics after each epoch
            self.save_metrics(run_dir)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, run_dir: str, name: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.log_data['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.log_data['train_loss'],
            'val_loss': self.log_data['val_loss'],
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(run_dir, f'{name}.pt')
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(run_dir, 'best.pt')
            torch.save(checkpoint, best_path)
    
    def save_metrics(self, run_dir: str):
        """Save metrics to JSON."""
        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.log_data['train_loss'] = checkpoint.get('train_loss', [])
        self.log_data['val_loss'] = checkpoint.get('val_loss', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

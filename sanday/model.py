"""
SANDAY ASR Model: Phoneme recognition with DREAM.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from dream import CoordinatedDREAMStack, DREAMConfig
from .phonemes import EnglishPhonemes


class SandayASR(nn.Module):
    """
    Speech Audio Neural Decoder with Acoustic Yield.
    
    Phoneme recognition model based on Coordinated DREAMStack.
    
    Architecture:
        Audio (80 mel bins)
            ↓
        Coordinated DREAMStack (3-4 layers)
            ↓
        CTC Projection (num_phonemes + 1 for blank)
            ↓
        Phoneme probabilities (with CTC loss)
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension (80 for mel spectrograms)
    hidden_dims : List[int]
        Hidden dimensions for each DREAM layer
    rank : int
        Fast weights rank
    num_phonemes : int
        Number of phoneme classes (including special tokens)
    dropout : float
        Dropout between DREAM layers
    use_coordination : bool
        Use coordinated DREAMStack with top-down modulation
    """
    
    def __init__(self,
                 input_dim: int = 80,
                 hidden_dims: list = None,
                 rank: int = 16,
                 num_phonemes: int = 45,  # 44 phonemes + blank
                 dropout: float = 0.1,
                 use_coordination: bool = True,
                 freeze_fast_weights: bool = False,  # NEW: For 2-stage training
                 ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]  # 3 layers
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.num_phonemes = num_phonemes
        self.num_layers = len(hidden_dims)
        self.freeze_fast_weights = freeze_fast_weights  # NEW
        
        # DREAM encoder (NEW: pass freeze_fast_weights)
        self.dream_stack = CoordinatedDREAMStack(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            rank=rank,
            dropout=dropout,
            use_hierarchical_tau=True,
            use_inter_layer_prediction=use_coordination,
            inter_layer_loss_weight=0.01,
            freeze_fast_weights=freeze_fast_weights,  # NEW
        )
        
        # CTC projection head
        self.ctc_head = nn.Linear(hidden_dims[-1], num_phonemes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection head weights."""
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.zeros_(self.ctc_head.bias)
    
    # ================================================================
    # 2-Stage Training Support
    # ================================================================
    
    def switch_to_pretraining(self, use_mse: bool = True):
        """
        Stage 1: Pre-training with frozen fast weights.
        
        Use reconstruction loss (MSE) to train slow weights.
        """
        self.dream_stack.switch_to_pretraining()
        
        # For Stage 1, use reconstruction loss
        if use_mse:
            self.reconstruction_head = nn.Linear(
                self.hidden_dims[-1],
                self.input_dim
            ).to(next(self.parameters()).device)
        
        print(f"[SandayASR] Stage 1: PRE-TRAINING")
        print(f"  - Fast weights: FROZEN")
        print(f"  - Loss: Reconstruction (MSE)")
    
    def switch_to_adaptation(self):
        """
        Stage 2: Adaptation training with active fast weights.
        
        Use CTC loss for phoneme recognition.
        """
        self.dream_stack.switch_to_adaptation()
        
        # Remove reconstruction head if exists
        if hasattr(self, 'reconstruction_head'):
            del self.reconstruction_head
        
        print(f"[SandayASR] Stage 2: ADAPTATION")
        print(f"  - Fast weights: ACTIVE")
        print(f"  - Loss: CTC (phoneme recognition)")
    
    def is_pretraining(self) -> bool:
        """Check if in pre-training mode."""
        return self.dream_stack.freeze_fast_weights
    
    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                return_states: bool = False,
                stage: str = 'adaptation',  # NEW: 'pretrain' or 'adaptation'
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input audio features (batch, time, input_dim)
        lengths : torch.Tensor, optional
            Sequence lengths (batch,)
        return_states : bool
            If True, return DREAM states for hierarchy analysis
        stage : str
            'pretrain' → return reconstruction
            'adaptation' → return CTC logits
        
        Returns
        -------
        outputs : Dict[str, torch.Tensor]
            Dictionary with:
            - Stage 1 (pretrain): 'recon', 'recon_loss'
            - Stage 2 (adaptation): 'ctc_logits', 'log_probs'
            - Both: 'states' (if return_states=True)
        """
        batch_size, time_steps, _ = x.shape
        device = x.device
        
        # Initialize states
        states = self.dream_stack.init_states(batch_size, device=device)
        
        # Process through DREAM stack
        all_outputs = []
        
        for t in range(time_steps):
            x_t = x[:, t, :]
            current_input = x_t
            
            for i, cell in enumerate(self.dream_stack.cells):
                h_new, states.layer_states[i], _, _ = cell(
                    current_input,
                    states.layer_states[i]
                )
                
                if i < self.num_layers - 1:
                    current_input = h_new
            
            all_outputs.append(h_new.unsqueeze(1))
        
        # Concatenate outputs
        hidden = torch.cat(all_outputs, dim=1)  # (batch, time, hidden_dim)
        
        outputs = {}
        
        # Stage 1: Reconstruction
        if stage == 'pretrain' or self.is_pretraining():
            if hasattr(self, 'reconstruction_head'):
                recon = self.reconstruction_head(hidden)
            else:
                recon = self.dream_stack.output_projection(hidden)
            
            recon_loss = nn.functional.mse_loss(recon, x)
            
            outputs['recon'] = recon
            outputs['recon_loss'] = recon_loss
            outputs['hidden'] = hidden
        
        # Stage 2: CTC
        else:
            ctc_logits = self.ctc_head(hidden)
            log_probs = torch.log_softmax(ctc_logits, dim=-1)
            
            outputs['ctc_logits'] = ctc_logits
            outputs['log_probs'] = log_probs
            outputs['hidden'] = hidden
        
        if return_states:
            outputs['states'] = states
        
        return outputs
    
    def decode(self,
               x: torch.Tensor,
               phonemes: EnglishPhonemes,
               blank_id: int = 1,
               ) -> Tuple[str, str]:
        """
        Decode audio to phonemes (greedy CTC decoding).
        
        Parameters
        ----------
        x : torch.Tensor
            Input audio features (1, time, input_dim)
        phonemes : EnglishPhonemes
            Phoneme converter
        blank_id : int
            ID of blank token
        
        Returns
        -------
        predicted_phonemes : str
            Decoded phoneme sequence
        aligned_text : str
            Phonemes as space-separated string
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self(x, return_states=False)
        
        # Greedy decoding
        log_probs = outputs['log_probs'][0]  # (time, num_phonemes)
        pred_ids = log_probs.argmax(dim=-1).cpu().tolist()
        
        # Remove blanks and duplicates (CTC decoding)
        prev_id = blank_id
        result = []
        for curr_id in pred_ids:
            if curr_id != blank_id and curr_id != prev_id:
                result.append(curr_id)
            prev_id = curr_id
        
        # Convert to phonemes
        phoneme_list = phonemes.ids_to_phonemes(result)
        
        return phoneme_list, ' '.join(phoneme_list)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create model
    model = SandayASR(
        input_dim=80,
        hidden_dims=[256, 256, 256],
        rank=16,
        num_phonemes=45,
        use_coordination=True,
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 100, 80)  # (batch, time, features)
    outputs = model(x)
    
    print(f"CTC logits shape: {outputs['ctc_logits'].shape}")
    print(f"Log probs shape: {outputs['log_probs'].shape}")

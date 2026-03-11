"""
Coordinated DREAMStack — Hierarchical Predictive Coding.

Key features:
1. Working Top-Down Modulation — реально влияет на пластичность
2. Hierarchical Tau — верхние слои медленнее (интегрируют дольше)
3. Inter-Layer Prediction — предсказания между слоями + loss

Architecture:
```
Input → [Layer 0] → h₀ → [Layer 1] → h₁ → [Layer 2] → h₂
          ↑  ↓         ↑  ↓         ↑  ↓
       pred₀  mod₁  pred₁  mod₂  pred₂  mod₃
          │              │              │
          └──── error ───┴──── error ───┘
                    ↓
            inter_layer_loss
```

Usage:
    from dream.layer_coordinated import CoordinatedDREAMStack
    
    model = CoordinatedDREAMStack(
        input_dim=80,
        hidden_dims=[128, 128, 128],
        rank=16,
        use_hierarchical_tau=True,
        use_inter_layer_prediction=True
    )
    
    output, states, losses = model(x, return_losses=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from dream import DREAMConfig, DREAMCell, DREAMState


@dataclass
class CoordinatedState:
    """State for coordinated DREAMStack."""
    layer_states: List[DREAMState]
    predictions: List[torch.Tensor]
    modulations: List[torch.Tensor]


class CoordinatedDREAMCell(DREAMCell):
    """
    DREAM Cell with coordination support.

    Adds:
    - Top-down modulation input (влияет на пластичность)
    - Layer prediction output
    - Hierarchical tau scaling
    """

    def __init__(
        self,
        config: DREAMConfig,
        layer_idx: int = 0,
        num_layers: int = 1,
        use_hierarchical_tau: bool = True
    ):
        super().__init__(config, use_coordination=True)
        
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.use_hierarchical_tau = use_hierarchical_tau
        
        # Hierarchical tau: upper layers have larger tau (slower adaptation)
        # Layer 0: factor=1.0, Layer 1: factor=1.5, Layer 2: factor=2.0, Layer 3: factor=2.5
        if use_hierarchical_tau:
            self.tau_depth_factor = 1.0 + 0.5 * layer_idx  # 1.0, 1.5, 2.0, 2.5
        else:
            self.tau_depth_factor = 1.0
        
        # Prediction head: predict lower layer activity (with normalization)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Modulation head: generate top-down modulation in [0, 1]
        self.modulation_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def compute_ltc_update(
        self,
        h_prev: torch.Tensor,
        u_eff: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        LTC update with hierarchical tau.

        Upper layers have larger tau → slower adaptation → longer integration.
        """
        if self.tau_sys.item() < 0.01:
            return torch.tanh(u_eff)

        # Apply depth factor to tau
        tau_base = self.tau_sys * self.tau_depth_factor
        
        # Dynamic tau modulated by surprise
        tau = tau_base / (1.0 + surprise * self.tau_surprise_scale)
        tau = torch.clamp(tau, 0.01, 50.0)

        # Euler integration
        h_target = torch.tanh(u_eff)
        dt_over_tau = self.dt / (tau.unsqueeze(1) + self.dt)
        dt_over_tau = torch.clamp(dt_over_tau, 0.01, 0.5)

        h_new = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target
        return h_new

    def forward(
        self,
        x: torch.Tensor,
        state: DREAMState,
        modulation_from_above: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, DREAMState, torch.Tensor, torch.Tensor]:
        """
        Forward pass with coordination.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, input_dim)
        state : DREAMState
            Cell state
        modulation_from_above : torch.Tensor, optional
            Top-down modulation from higher layer (batch, hidden_dim)

        Returns
        -------
        h_new : torch.Tensor
            New hidden state
        state : DREAMState
            Updated state
        prediction : torch.Tensor
            Prediction for lower layer
        modulation : torch.Tensor
            Top-down modulation for lower layer
        """
        batch_size = x.shape[0]

        # ================================================================
        # 1. Predictive Coding
        # ================================================================
        x_pred = torch.tanh(state.h @ self.C)
        error = x - x_pred

        # ================================================================
        # 2. Surprise Gate WITH top-down modulation
        # ================================================================
        surprise, error_norm = self.compute_surprise(error, state, modulation_from_above)

        # ================================================================
        # 3. Fast Weights Update WITH modulation-enhanced plasticity
        # ================================================================
        # Modulation enhances plasticity: higher modulation → faster learning
        # But use gentle modulation to avoid destabilizing training
        if modulation_from_above is not None and self.use_coordination:
            # Modulation is in [0, 1], apply as gentle multiplicative factor
            # Base plasticity = 1.0, modulation scales it by ±20%
            plasticity_boost = 1.0 + 0.2 * (modulation_from_above.mean() - 0.5)
            effective_eta = self.eta * plasticity_boost
        else:
            effective_eta = self.eta

        # Update with effective plasticity
        state = self._update_fast_weights_with_eta(
            state.h, error, surprise, state, effective_eta
        )

        # ================================================================
        # 4. Gain Modulation
        # ================================================================
        gain = 1.0 + self.kappa * surprise.unsqueeze(1)
        base_effect = x @ self.B_base
        u_eff = gain * base_effect

        # Fast weights contribution
        fast_effect = torch.bmm(state.U, self.V.T.unsqueeze(0).expand(batch_size, -1, -1))
        fast_effect = torch.bmm(fast_effect, x.unsqueeze(2)).squeeze(2)
        u_eff = u_eff + fast_effect * 0.1

        # ================================================================
        # 5. LTC Update WITH hierarchical tau
        # ================================================================
        h_ltc = self.compute_ltc_update(state.h, u_eff, surprise)
        error_injection = error @ self.W
        h_new = h_ltc + error_injection
        h_new = h_new * 0.99 + state.h * 0.01

        # ================================================================
        # 6. Update Statistics
        # ================================================================
        state.error_mean = (1 - self.beta) * state.error_mean + self.beta * error
        state.error_var = (1 - self.beta) * state.error_var + self.beta * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        # ================================================================
        # 7. Generate Prediction & Modulation
        # ================================================================
        # Prediction for lower layer
        prediction = self.prediction_head(h_new)
        
        # Top-down modulation for lower layer
        modulation = self.modulation_head(h_new)

        return h_new, state, prediction, modulation

    def _update_fast_weights_with_eta(
        self,
        h_prev: torch.Tensor,
        error: torch.Tensor,
        surprise: torch.Tensor,
        state: DREAMState,
        effective_eta: torch.Tensor
    ) -> DREAMState:
        """Update fast weights with custom eta."""
        if self.freeze_fast_weights:
            return state

        batch_size = h_prev.shape[0]

        # Hebbian term
        eV = error @ self.V
        hebbian = state.h.unsqueeze(2) * eV.unsqueeze(1)

        # Plasticity modulation with effective eta
        plasticity = effective_eta.unsqueeze(0) * surprise.unsqueeze(1)
        plasticity = plasticity.unsqueeze(2)

        # Forgetting term
        forgetting = -self.forgetting_rate * (state.U - state.U_target)

        # Full update
        dU = forgetting + plasticity * hebbian
        U_new = state.U + dU * self.dt

        # Normalization
        U_norm = U_new.norm(dim=(1, 2), keepdim=True)
        scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=2.0)
        state.U = U_new * scale

        return state


class CoordinatedDREAMStack(nn.Module):
    """
    Coordinated DREAMStack with hierarchical predictive coding.

    Features:
    - Top-down modulation that реально влияет на пластичность
    - Hierarchical tau (верхние слои медленнее)
    - Inter-layer prediction loss
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dims: List[int] = None,
        rank: int = 16,
        dropout: float = 0.1,
        use_hierarchical_tau: bool = True,
        use_inter_layer_prediction: bool = True,
        inter_layer_loss_weight: float = 0.01,
        freeze_fast_weights: bool = False,  # NEW: Support for 2-stage training
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.rank = rank
        self.num_layers = len(hidden_dims)
        self.use_hierarchical_tau = use_hierarchical_tau
        self.use_inter_layer_prediction = use_inter_layer_prediction
        self.inter_layer_loss_weight = inter_layer_loss_weight
        self.freeze_fast_weights = freeze_fast_weights  # NEW

        # Create coordinated cells
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim_i = input_dim if i == 0 else hidden_dims[i-1]
            config = DREAMConfig(
                input_dim=input_dim_i,
                hidden_dim=hidden_dims[i],
                rank=rank,
                use_coordination=True
            )
            cell = CoordinatedDREAMCell(
                config,
                layer_idx=i,
                num_layers=self.num_layers,
                use_hierarchical_tau=use_hierarchical_tau
            )
            # NEW: Apply freeze_fast_weights to all cells
            cell.freeze_fast_weights = freeze_fast_weights
            self.cells.append(cell)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection (to input dim for reconstruction)
        self.output_projection = nn.Linear(hidden_dims[-1], input_dim)

    # ================================================================
    # NEW: 2-Stage Training Support
    # ================================================================
    
    def set_fast_weights_mode(self, freeze: bool):
        """
        Freeze/unfreeze fast weights in ALL layers.
        
        Parameters
        ----------
        freeze : bool
            True  = Stage 1 (pre-training, fast weights frozen)
            False = Stage 2 (adaptation, fast weights active)
        """
        self.freeze_fast_weights = freeze
        for cell in self.cells:
            cell.freeze_fast_weights = freeze
    
    def train(self, mode: bool = True):
        """
        Set training mode and auto-freeze fast weights.
        
        Stage 1 (pre-training): model.train() → fast weights FROZEN
        Stage 2 (adaptation):   model.eval()  → fast weights ACTIVE
        """
        super().train(mode)
        # Auto-freeze fast weights during training (Stage 1)
        # Unfreeze during eval (Stage 2/Production)
        self.set_fast_weights_mode(freeze=mode)
        return self
    
    def switch_to_adaptation(self):
        """
        Switch from Stage 1 (pre-training) to Stage 2 (adaptation).
        
        Call this after pre-training to enable fast weights plasticity.
        """
        self.set_fast_weights_mode(freeze=False)
        self.eval()  # Set to eval mode
        print(f"[CoordinatedDREAMStack] Switched to ADAPTATION mode")
        print(f"  - Fast weights: UNFROZEN (active plasticity)")
        print(f"  - Mode: eval (inference with adaptation)")
    
    def switch_to_pretraining(self):
        """
        Switch to Stage 1 (pre-training) mode.
        
        Call this to freeze fast weights and train slow weights.
        """
        self.set_fast_weights_mode(freeze=True)
        self.train()  # Set to train mode
        print(f"[CoordinatedDREAMStack] Switched to PRE-TRAINING mode")
        print(f"  - Fast weights: FROZEN (static base)")
        print(f"  - Mode: train (slow weights learning)")

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> CoordinatedState:
        """Initialize states for all layers."""
        layer_states = [
            cell.init_state(batch_size, device=device)
            for cell in self.cells
        ]
        
        # Initialize predictions and modulations as zeros
        predictions = [
            torch.zeros(batch_size, h, device=device)
            for h in self.hidden_dims
        ]
        modulations = [
            torch.zeros(batch_size, h, device=device)
            for h in self.hidden_dims
        ]

        return CoordinatedState(
            layer_states=layer_states,
            predictions=predictions,
            modulations=modulations
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[CoordinatedState] = None,
        return_losses: bool = False
    ) -> Tuple[torch.Tensor, CoordinatedState, Optional[Dict]]:
        """
        Forward pass through coordinated stack.

        Processing:
        - Bottom-up: process input through layers
        - Top-down modulation from previous timestep (stored in states)

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        states : CoordinatedState, optional
            Coordinated states
        return_losses : bool
            Return inter-layer prediction losses

        Returns
        -------
        output : torch.Tensor
            Output (batch, time, input_dim)
        states : CoordinatedState
            Updated states
        losses : dict, optional
            Prediction losses
        """
        batch_size, time_steps, _ = x.shape
        device = x.device

        if states is None:
            states = self.init_states(batch_size, device)

        losses = {'reconstruction': 0.0, 'inter_layer': 0.0} if return_losses else None

        # Process sequence
        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch, input_dim)

            # ================================================================
            # BOTTOM-UP PASS
            # ================================================================
            layer_outputs = []
            current_input = x_t

            for i, cell in enumerate(self.cells):
                # Get modulation from layer above (None for top layer)
                # Use modulation from PREVIOUS timestep (already detached)
                modulation = states.modulations[i+1] if i < self.num_layers - 1 else None

                # Forward through cell
                h_new, states.layer_states[i], prediction, modulation_out = cell(
                    current_input,
                    states.layer_states[i],
                    modulation
                )

                layer_outputs.append(h_new)
                
                # Store prediction and modulation for NEXT timestep
                # Detach to prevent second-order gradients across timesteps
                states.predictions[i] = prediction.detach()
                states.modulations[i] = modulation_out.detach()

                # Compute inter-layer prediction loss
                if self.use_inter_layer_prediction and return_losses and i > 0:
                    # Prediction from layer i for layer i-1
                    # Use detached prediction from previous timestep
                    pred_lower = states.predictions[i]
                    # Actual output from layer i-1
                    actual_lower = layer_outputs[i-1]
                    # Prediction error (scaled down to not dominate reconstruction)
                    inter_error = F.mse_loss(pred_lower, actual_lower)
                    # Normalize by dimension to prevent large values
                    inter_error = inter_error / self.hidden_dims[i]
                    losses['inter_layer'] = losses['inter_layer'] + inter_error

                # Prepare input for next layer
                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

            # Reconstruction loss (top layer output → input)
            if return_losses:
                recon = self.output_projection(layer_outputs[-1])
                losses['reconstruction'] = F.mse_loss(recon, x_t)

        # Final output projection
        final_output = layer_outputs[-1]
        output = self.output_projection(final_output)

        return output, states, losses

    def forward_sequence(
        self,
        x: torch.Tensor,
        states: Optional[CoordinatedState] = None,
        return_all: bool = True
    ) -> Tuple[torch.Tensor, CoordinatedState]:
        """Process full sequence (compatibility with existing code)."""
        output, states, _ = self.forward(x, states, return_losses=False)
        
        if return_all:
            return output, states
        else:
            return output[:, -1, :], states

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UncoordinatedDREAMStack(nn.Module):
    """
    Uncoordinated DREAM Stack (baseline for comparison).

    Standard stack without top-down modulation or inter-layer prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rank: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Create layers WITHOUT coordination
        self.layers = nn.ModuleList()

        for i, h in enumerate(hidden_dims):
            input_dim_i = input_dim if i == 0 else hidden_dims[i-1]
            config = DREAMConfig(
                input_dim=input_dim_i,
                hidden_dim=h,
                rank=rank,
                use_coordination=False
            )
            self.layers.append(DREAMCell(config))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> List[DREAMState]:
        """Initialize states for all layers."""
        return [layer.init_state(batch_size, device=device) for layer in self.layers]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[DREAMState]] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, List[DREAMState]]:
        """Standard forward pass without coordination."""
        batch_size, time_steps, _ = x.shape

        if states is None:
            states = self.init_states(batch_size, device=x.device)

        if return_all:
            all_outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :]
            current_input = x_t

            for i, layer in enumerate(self.layers):
                h_new, states[i] = layer(current_input, states[i])

                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

            if return_all:
                all_outputs.append(h_new.unsqueeze(1))

        if return_all:
            output = torch.cat(all_outputs, dim=1)
        else:
            output = h_new

        return output, states

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

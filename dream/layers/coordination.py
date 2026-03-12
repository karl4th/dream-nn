"""
Coordination Module - Hierarchical Predictive Coding.

Optional module for coordinated DREAM stack with:
- Top-down modulation (влияет на пластичность)
- Hierarchical tau (верхние слои медленнее)
- Inter-layer prediction + loss

Usage:
    from dream.layers.coordination import CoordinatedDREAMStack

    model = CoordinatedDREAMStack(
        input_dim=80,
        hidden_dims=[128, 128, 128],
        rank=16,
        use_hierarchical_tau=True,
        use_inter_layer_prediction=True
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..config import DREAMConfig
from ..state import DREAMState
from .predictive_coding import PredictiveCoding
from .surprise_gate import SurpriseGate
from .fast_weights import FastWeights
from .ltc import LiquidTimeConstants
from .sleep_consolidation import SleepConsolidation


@dataclass
class CoordinatedState:
    """State for coordinated DREAMStack."""
    layer_states: List[DREAMState]
    predictions: List[torch.Tensor]
    modulations: List[torch.Tensor]


class CoordinatedDREAMCell(nn.Module):
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
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.use_hierarchical_tau = use_hierarchical_tau
        self.freeze_fast_weights = False

        # ================================================================
        # Block 1: Predictive Coding
        # ================================================================
        self.predictive_coding = PredictiveCoding(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim
        )

        # ================================================================
        # Block 2: Surprise Gate
        # ================================================================
        self.surprise_gate = SurpriseGate(
            hidden_dim=config.hidden_dim,
            base_threshold=config.base_threshold,
            entropy_influence=config.entropy_influence,
            surprise_temperature=config.surprise_temperature,
            kappa=config.kappa
        )

        # ================================================================
        # Block 3: Fast Weights
        # ================================================================
        self.use_fast_weights = config.use_fast_weights
        self.fast_weights = FastWeights(
            hidden_dim=config.hidden_dim,
            input_dim=config.input_dim,
            rank=config.rank,
            forgetting_rate=config.forgetting_rate,
            base_plasticity=config.base_plasticity,
            target_norm=config.target_norm,
            time_step=config.time_step,
            freeze_fast_weights=False
        )

        # ================================================================
        # Block 4: Liquid Time-Constants with hierarchical tau
        # ================================================================
        self.use_ltc = config.use_ltc

        # Hierarchical tau: upper layers have larger tau (slower adaptation)
        if use_hierarchical_tau:
            self.tau_depth_factor = 1.0 + 0.5 * layer_idx  # 1.0, 1.5, 2.0, 2.5
        else:
            self.tau_depth_factor = 1.0

        self.ltc = LiquidTimeConstants(
            ltc_tau_sys=config.ltc_tau_sys,
            ltc_surprise_scale=config.ltc_surprise_scale,
            time_step=config.time_step,
            ltc_enabled=config.use_ltc
        )

        # ================================================================
        # Block 5: Sleep Consolidation
        # ================================================================
        self.use_sleep = config.use_sleep
        self.sleep = SleepConsolidation(
            sleep_rate=config.sleep_rate,
            min_surprise_for_sleep=config.min_surprise_for_sleep,
            target_norm=config.target_norm
        )

        # ================================================================
        # Coordination: Prediction & Modulation heads
        # ================================================================
        # Prediction head: predict lower layer activity
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
            nn.Sigmoid()
        )

        # ================================================================
        # Smoothing Parameters
        # ================================================================
        self.register_buffer('beta', torch.tensor(config.error_smoothing))
        self.register_buffer('beta_s', torch.tensor(config.surprise_smoothing))

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> DREAMState:
        """Initialize cell state."""
        return DREAMState.init_from_config(
            self.config, batch_size, device, dtype
        )

    def compute_ltc_with_hierarchical_tau(
        self,
        h_prev: torch.Tensor,
        u_eff: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """LTC update with hierarchical tau."""
        if not self.use_ltc or self.ltc.tau_sys.item() < 0.01:
            return torch.tanh(u_eff)

        # Apply depth factor to tau
        tau_base = self.ltc.tau_sys * self.tau_depth_factor

        # Dynamic tau modulated by surprise
        tau = tau_base / (1.0 + surprise * self.ltc.tau_surprise_scale)
        tau = torch.clamp(tau, 0.01, 50.0)

        # Euler integration
        h_target = torch.tanh(u_eff)
        dt_over_tau = self.ltc.time_step / (tau.unsqueeze(1) + self.ltc.time_step)
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
        x_pred, error = self.predictive_coding(x, state.h)

        # ================================================================
        # 2. Surprise Gate WITH top-down modulation
        # ================================================================
        surprise, error_norm, gain, state.surprise_mu, state.surprise_sigma = self.surprise_gate(
            error, state.error_var, state.error_mean, state.surprise_mu, state.surprise_sigma
        )

        # Apply modulation to surprise (makes layer more sensitive)
        if modulation_from_above is not None:
            modulation_strength = modulation_from_above.mean(dim=-1)  # (batch,)
            surprise = surprise * (1.0 + 0.2 * modulation_strength)

        # ================================================================
        # 3. Fast Weights Update
        # ================================================================
        if self.use_fast_weights:
            # Modulation enhances plasticity
            if modulation_from_above is not None:
                plasticity_boost = 1.0 + 0.2 * (modulation_from_above.mean() - 0.5)
                effective_eta = self.fast_weights.eta * plasticity_boost
            else:
                effective_eta = self.fast_weights.eta

            state.U = self.fast_weights.update(
                state.h, error, surprise, state.U, state.U_target
            )

        # ================================================================
        # 4. Effective Input Projection
        # ================================================================
        base_effect = self.predictive_coding.project_input(x)
        u_eff = gain * base_effect

        if self.use_fast_weights:
            fast_effect = self.fast_weights.compute_fast_effect(
                state.U, self.fast_weights.V, x
            )
            u_eff = u_eff + fast_effect * 0.1

        # ================================================================
        # 5. State Update with hierarchical LTC
        # ================================================================
        h_ltc = self.compute_ltc_with_hierarchical_tau(state.h, u_eff, surprise)
        error_injection = self.predictive_coding.inject_error(error)
        h_new = h_ltc + error_injection
        h_new = h_new * 0.99 + state.h * 0.01

        # ================================================================
        # 6. Update Statistics
        # ================================================================
        state.error_mean = (1 - self.beta) * state.error_mean + self.beta * error
        state.error_var = (1 - self.beta) * state.error_var + self.beta * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        # ================================================================
        # 7. Sleep Consolidation
        # ================================================================
        if self.use_sleep:
            state.U_target, sleep_triggered = self.sleep(
                state.U, state.U_target, state.avg_surprise, state.error_mean
            )

        # ================================================================
        # 8. Generate Prediction & Modulation
        # ================================================================
        prediction = self.prediction_head(h_new)
        modulation = self.modulation_head(h_new)

        return h_new, state, prediction, modulation

    def set_fast_weights_mode(self, freeze: bool):
        """Set fast weights training mode."""
        self.freeze_fast_weights = freeze
        if self.use_fast_weights:
            self.fast_weights.freeze_fast_weights = freeze


class CoordinatedDREAMStack(nn.Module):
    """
    Coordinated DREAMStack with hierarchical predictive coding.

    Features:
    - Top-down modulation (влияет на пластичность)
    - Hierarchical tau (верхние слои медленнее)
    - Inter-layer prediction loss

    Usage:
        model = CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128, 128],
            rank=16,
            use_hierarchical_tau=True,
            use_inter_layer_prediction=True
        )
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
        freeze_fast_weights: bool = False,
        **config_kwargs
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
        self.freeze_fast_weights = freeze_fast_weights

        # Create coordinated cells
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim_i = input_dim if i == 0 else hidden_dims[i-1]
            config = DREAMConfig(
                input_dim=input_dim_i,
                hidden_dim=hidden_dims[i],
                rank=rank,
                **config_kwargs
            )
            cell = CoordinatedDREAMCell(
                config,
                layer_idx=i,
                num_layers=self.num_layers,
                use_hierarchical_tau=use_hierarchical_tau
            )
            cell.freeze_fast_weights = freeze_fast_weights
            self.cells.append(cell)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection
        self.output_projection = nn.Linear(hidden_dims[-1], input_dim)

    def set_fast_weights_mode(self, freeze: bool):
        """Freeze/unfreeze fast weights in ALL layers."""
        self.freeze_fast_weights = freeze
        for cell in self.cells:
            cell.set_fast_weights_mode(freeze)

    def train(self, mode: bool = True):
        """Set training mode and auto-freeze fast weights."""
        super().train(mode)
        self.set_fast_weights_mode(freeze=mode)
        return self

    def switch_to_adaptation(self):
        """Switch from pre-training to adaptation mode."""
        self.set_fast_weights_mode(freeze=False)
        self.eval()
        print(f"[CoordinatedDREAMStack] ADAPTATION mode")

    def switch_to_pretraining(self):
        """Switch to pre-training mode."""
        self.set_fast_weights_mode(freeze=True)
        self.train()
        print(f"[CoordinatedDREAMStack] PRE-TRAINING mode")

    def init_states(self, batch_size: int, device: Optional[torch.device] = None) -> CoordinatedState:
        """Initialize states for all layers."""
        layer_states = [
            cell.init_state(batch_size, device=device)
            for cell in self.cells
        ]

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
        - Top-down modulation from previous timestep
        """
        batch_size, time_steps, _ = x.shape
        device = x.device

        if states is None:
            states = self.init_states(batch_size, device)

        losses = {'reconstruction': 0.0, 'inter_layer': 0.0} if return_losses else None

        for t in range(time_steps):
            x_t = x[:, t, :]
            layer_outputs = []
            current_input = x_t

            for i, cell in enumerate(self.cells):
                modulation = states.modulations[i+1] if i < self.num_layers - 1 else None

                h_new, states.layer_states[i], prediction, modulation_out = cell(
                    current_input,
                    states.layer_states[i],
                    modulation
                )

                layer_outputs.append(h_new)
                states.predictions[i] = prediction.detach()
                states.modulations[i] = modulation_out.detach()

                if self.use_inter_layer_prediction and return_losses and i > 0:
                    pred_lower = states.predictions[i]
                    actual_lower = layer_outputs[i-1]
                    inter_error = F.mse_loss(pred_lower, actual_lower)
                    inter_error = inter_error / self.hidden_dims[i]
                    losses['inter_layer'] = losses['inter_layer'] + inter_error

                if i < self.num_layers - 1:
                    current_input = h_new
                    if self.dropout is not None:
                        current_input = self.dropout(current_input)

            if return_losses:
                recon = self.output_projection(layer_outputs[-1])
                losses['reconstruction'] = F.mse_loss(recon, x_t)

        final_output = layer_outputs[-1]
        output = self.output_projection(final_output)

        return output, states, losses

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

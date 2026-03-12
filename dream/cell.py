"""
DREAM Cell - Modular implementation.

Dynamic Recall and Elastic Adaptive Memory with pluggable blocks:
- Predictive Coding (required)
- Surprise Gate (required)
- Fast Weights (optional)
- Liquid Time-Constants (optional)
- Sleep Consolidation (optional)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .config import DREAMConfig
from .state import DREAMState
from .layers import PredictiveCoding, SurpriseGate, FastWeights, LiquidTimeConstants, SleepConsolidation


class DREAMCell(nn.Module):
    """
    DREAM (Dynamic Recall and Elastic Adaptive Memory) cell.

    A continuous-time RNN cell with modular blocks:
    - Predictive Coding: prediction and error computation
    - Surprise Gate: surprise-driven modulation
    - Fast Weights: Hebbian learning with low-rank decomposition
    - LTC: adaptive integration time constants
    - Sleep: memory consolidation

    Parameters
    ----------
    config : DREAMConfig
        Model configuration
    freeze_fast_weights : bool, default=False
        If True, fast weights are frozen during training

    Examples
    --------
    >>> from dream import DREAMConfig, DREAMCell
    >>> config = DREAMConfig(input_dim=39, hidden_dim=256)
    >>> cell = DREAMCell(config)
    >>> state = cell.init_state(batch_size=32)
    >>> h, state = cell(x, state)
    """

    def __init__(self, config: DREAMConfig, freeze_fast_weights: bool = False):
        super().__init__()
        self.config = config
        self.freeze_fast_weights = freeze_fast_weights

        # ================================================================
        # Block 1: Predictive Coding (required)
        # ================================================================
        self.predictive_coding = PredictiveCoding(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim
        )

        # ================================================================
        # Block 2: Surprise Gate (required)
        # ================================================================
        self.surprise_gate = SurpriseGate(
            hidden_dim=config.hidden_dim,
            base_threshold=config.base_threshold,
            entropy_influence=config.entropy_influence,
            surprise_temperature=config.surprise_temperature,
            kappa=config.kappa
        )

        # ================================================================
        # Block 3: Fast Weights (optional)
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
            freeze_fast_weights=freeze_fast_weights
        )

        # ================================================================
        # Block 4: Liquid Time-Constants (optional)
        # ================================================================
        self.use_ltc = config.use_ltc
        self.ltc = LiquidTimeConstants(
            ltc_tau_sys=config.ltc_tau_sys,
            ltc_surprise_scale=config.ltc_surprise_scale,
            time_step=config.time_step,
            ltc_enabled=config.use_ltc
        )

        # ================================================================
        # Block 5: Sleep Consolidation (optional)
        # ================================================================
        self.use_sleep = config.use_sleep
        self.sleep = SleepConsolidation(
            sleep_rate=config.sleep_rate,
            min_surprise_for_sleep=config.min_surprise_for_sleep,
            target_norm=config.target_norm
        )

        # ================================================================
        # Smoothing Parameters (for statistics)
        # ================================================================
        self.register_buffer('beta', torch.tensor(config.error_smoothing))
        self.register_buffer('beta_s', torch.tensor(config.surprise_smoothing))

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> DREAMState:
        """
        Initialize cell state.

        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device, optional
            Device for tensors
        dtype : torch.dtype, optional
            Data type for tensors

        Returns
        -------
        DREAMState
            Initialized state
        """
        return DREAMState.init_from_config(
            self.config, batch_size, device, dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        state: DREAMState
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Forward pass of DREAM cell.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, input_dim)
        state : DREAMState
            Current state

        Returns
        -------
        h_new : torch.Tensor
            New hidden state (batch, hidden_dim)
        state : DREAMState
            Updated state
        """
        batch_size = x.shape[0]

        # ================================================================
        # 1. Predictive Coding
        # ================================================================
        x_pred, error = self.predictive_coding(x, state.h)

        # ================================================================
        # 2. Surprise Gate
        # ================================================================
        surprise, error_norm, gain, state.surprise_mu, state.surprise_sigma = self.surprise_gate(
            error, state.error_var, state.error_mean, state.surprise_mu, state.surprise_sigma
        )

        # ================================================================
        # 3. Fast Weights Update
        # ================================================================
        if self.use_fast_weights:
            state.U = self.fast_weights.update(
                state.h, error, surprise, state.U, state.U_target
            )

        # ================================================================
        # 4. Effective Input Projection
        # ================================================================
        # Base projection with gain modulation
        base_effect = self.predictive_coding.project_input(x)
        u_eff = gain * base_effect

        # Add fast weights contribution
        if self.use_fast_weights:
            fast_effect = self.fast_weights.compute_fast_effect(
                state.U, self.fast_weights.V, x
            )
            u_eff = u_eff + fast_effect * 0.1

        # ================================================================
        # 5. State Update with LTC
        # ================================================================
        h_ltc = self.ltc(state.h, u_eff, surprise)

        # Error injection
        error_injection = self.predictive_coding.inject_error(error)

        # Combine
        h_new = h_ltc + error_injection

        # Stability: mild leaky integration
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

        return h_new, state

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        state: Optional[DREAMState] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Process full sequence through cell.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input sequence (batch, time, input_dim)
        state : DREAMState, optional
            Initial state
        return_all : bool
            If True, return all hidden states

        Returns
        -------
        output : torch.Tensor
            If return_all: (batch, time, hidden_dim)
            Else: (batch, hidden_dim) - final state
        state : DREAMState
            Final state
        """
        batch_size, time_steps, _ = x_seq.shape

        if state is None:
            state = self.init_state(batch_size, device=x_seq.device, dtype=x_seq.dtype)

        if return_all:
            all_h = []

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            h, state = self(x_t, state)

            if return_all:
                all_h.append(h.unsqueeze(1))

        if return_all:
            output = torch.cat(all_h, dim=1)
        else:
            output = h

        return output, state

    def set_fast_weights_mode(self, freeze: bool):
        """
        Set fast weights training mode.

        Parameters
        ----------
        freeze : bool
            True  = Static base training (fast weights frozen)
            False = Adaptation/Inference (fast weights active)
        """
        self.freeze_fast_weights = freeze
        self.fast_weights.freeze_fast_weights = freeze

    def train(self, mode: bool = True):
        """
        Set training mode and auto-freeze fast weights.

        Training (mode=True): fast weights FROZEN
        Eval (mode=False): fast weights ACTIVE
        """
        super().train(mode)
        self.set_fast_weights_mode(freeze=mode)
        return self

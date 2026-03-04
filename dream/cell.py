"""DREAM Cell: Dynamic Recall and Elastic Adaptive Memory.

Implementation based on NNAI-S Architecture Specification v0.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .config import DREAMConfig
from .state import DREAMState


class DREAMCell(nn.Module):
    """
    DREAM (Dynamic Recall and Elastic Adaptive Memory) cell.

    A continuous-time RNN cell with:
    - Predictive coding with fast weights
    - Surprise-driven STDP plasticity
    - Liquid Time-Constants (LTC) for adaptive integration
    - Sleep consolidation for memory stabilization

    Architecture follows NNAI-S spec:

    1. Predictive Coding: x_hat = C @ h, e = x - x_hat
    2. Surprise Gate: S = sigmoid((||e|| - τ) / γ)
    3. Fast Weights: U updated via Hebbian learning modulated by S
    4. State Update: h_new = LTC(h, B_eff @ x) + W @ e
    5. Sleep: U_target consolidates U when surprise is high

    Examples
    --------
    >>> from dream import DREAMConfig, DREAMCell
    >>> config = DREAMConfig(input_dim=39, hidden_dim=256)
    >>> cell = DREAMCell(config)
    >>> state = cell.init_state(batch_size=32)
    >>> for t in range(sequence_length):
    ...     x = input_seq[:, t, :]
    ...     h, state = cell(x, state)
    """

    def __init__(self, config: DREAMConfig):
        super().__init__()
        self.config = config

        # ================================================================
        # BLOCK 1: Predictive Coding (Spec Section 2)
        # ================================================================
        # C: decoding matrix (hidden_dim -> input_dim)
        self.C = nn.Parameter(torch.randn(config.hidden_dim, config.input_dim) * 0.1)
        # W: error injection matrix (input_dim -> hidden_dim)
        self.W = nn.Parameter(torch.randn(config.input_dim, config.hidden_dim) * 0.1)
        # B_base: base input projection (input_dim -> hidden_dim)
        self.B_base = nn.Parameter(torch.randn(config.input_dim, config.hidden_dim) * 0.1)

        # ================================================================
        # BLOCK 2: Fast Weights (Spec Section 4)
        # ================================================================
        # V: fixed orthogonal sensory filter (input_dim, rank) per Spec 4.1
        V_init = torch.randn(config.input_dim, config.rank)
        # Orthogonalize via QR
        Q, _ = torch.linalg.qr(V_init)
        self.register_buffer('V', Q)  # Fixed, not learnable during inference

        # U: fast weights left factor (batch, hidden_dim, rank) - in state
        # eta: vector plasticity coefficient (hidden_dim,)
        self.eta = nn.Parameter(torch.ones(config.hidden_dim) * config.base_plasticity)

        # ================================================================
        # BLOCK 3: Surprise Gate (Spec Section 3)
        # ================================================================
        # tau_0: base threshold
        self.tau_0 = nn.Parameter(torch.tensor(config.base_threshold))
        # alpha: entropy influence
        self.alpha = nn.Parameter(torch.tensor(config.entropy_influence))
        # gamma: surprise temperature
        self.gamma = nn.Parameter(torch.tensor(config.surprise_temperature))
        # kappa: gain modulation for B
        self.kappa = nn.Parameter(torch.tensor(config.kappa))

        # ================================================================
        # BLOCK 4: Liquid Time-Constants (Spec Section 2.3)
        # ================================================================
        # tau_sys: base system time constant
        self.tau_sys = nn.Parameter(torch.tensor(config.ltc_tau_sys))
        # tau_surprise_scale: how much surprise affects tau
        self.tau_surprise_scale = config.ltc_surprise_scale

        # ================================================================
        # BLOCK 5: Sleep Consolidation (Spec Section 5)
        # ================================================================
        self.sleep_rate = config.sleep_rate
        self.S_min = config.min_surprise_for_sleep

        # ================================================================
        # Smoothing Parameters
        # ================================================================
        self.beta = config.error_smoothing  # for error statistics
        self.beta_s = config.surprise_smoothing  # for surprise statistics
        self.forgetting_rate = config.forgetting_rate
        self.target_norm = config.target_norm

        # Time step for continuous dynamics
        self.dt = config.time_step

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

    def compute_surprise(
        self,
        error: torch.Tensor,
        state: DREAMState
    ) -> torch.Tensor:
        """
        Compute surprise with adaptive threshold (Spec Section 3.2-3.3).

        S_t = sigmoid((||e|| - τ) / γ)
        τ = τ_0 * (1 + α * H)  where H = entropy from error variance

        Uses relative error norm for better noise detection.
        """
        batch_size = error.shape[0]
        eps = 1e-6

        # Error norm (batch,)
        error_norm = error.norm(dim=-1)

        # Entropy from error variance (Spec 3.3)
        # H = 0.5 * log(2πe * var)
        variance = state.error_var.mean(dim=-1)  # (batch,)
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + eps))
        entropy = torch.clamp(entropy, 0.0, 2.0)

        # Adaptive threshold (Spec 3.3)
        # Use running mean of error norm as baseline for comparison
        baseline_error = state.error_mean.norm(dim=-1) + eps
        
        # Relative surprise: how much does current error exceed expected?
        relative_error = error_norm / baseline_error
        
        # Threshold based on entropy (uncertainty)
        tau = 1.0 + self.alpha * entropy  # Base threshold around 1.0 (relative)
        
        # Surprise using relative error
        # gamma controls sensitivity: smaller = more sensitive
        surprise = torch.sigmoid((relative_error - tau) / (self.gamma * 2))

        return surprise, error_norm

    def update_fast_weights(
        self,
        h_prev: torch.Tensor,
        error: torch.Tensor,
        surprise: torch.Tensor,
        state: DREAMState
    ) -> None:
        """
        Update fast weights U via STDP (Spec Section 4.2).

        dU = -λ * (U - U_target) + (η * S) * (h_prev ⊗ error) @ V

        where ⊗ is outer product, @ is matrix multiplication.
        """
        batch_size = h_prev.shape[0]

        # Hebbian term: outer product projected onto V
        # (batch, hidden, 1) @ (batch, 1, input) = (batch, hidden, input)
        # Then @ V: (batch, hidden, input) @ (input, rank) -> WRONG!
        # V is (hidden, rank), so we need: (h ⊗ e) @ V
        # h: (batch, hidden), e: (batch, input)
        # outer: (batch, hidden, input)
        # We want: (batch, hidden, rank) = (batch, hidden, input) @ ??? 

        # Actually per spec: update = outer(h, e) @ V
        # But V is (hidden, rank), not (input, rank)
        # Let me re-read spec...

        # Spec 4.1: W_fast = U @ V.T where U: (d_state, rank), V: (d_model, rank)
        # Spec 4.2: update = outer(h, e) @ V
        # This means: outer(h,e) is (d_state, d_model), V is (d_model, rank)
        # So result is (d_state, rank) ✓

        # In our case: d_model = input_dim, d_state = hidden_dim
        # V should be (input_dim, rank) NOT (hidden_dim, rank)
        # Let me fix this...

        # Actually, looking at pseudocode in spec section 6:
        # update = torch.outer(h_prev, e_t) @ V  # V is (d_model, rank)
        # So V should match input dimension for sensory filtering

        # For now, let's use V as (input_dim, rank) to match spec
        # We'll need to change initialization too
        pass  # Will fix in full rewrite

    def compute_ltc_update(
        self,
        h_prev: torch.Tensor,
        u_eff: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hidden state update with LTC (Spec Section 2.3).

        dh/dt = (-h + tanh(u_eff)) / τ
        τ = τ_sys / (1 + S * scale)

        High surprise → small τ → fast updates
        Low surprise → large τ → slow integration
        """
        if self.tau_sys.item() < 0.01:
            # LTC disabled
            return torch.tanh(u_eff)

        # Dynamic time constant
        tau = self.tau_sys / (1.0 + surprise * self.tau_surprise_scale)
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
        state: DREAMState
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Forward pass of DREAM cell (Spec Section 6 step function).

        1. Predictive Coding: x_hat = C^T @ h, e = x - x_hat
        2. Surprise: S = sigmoid((||e|| - τ) / γ)
        3. Fast Weights: U += -λ(U - U_target) + (η*S) * (h ⊗ e) @ V
        4. Gain modulation: B_eff = (1 + κ*S) * B_base
        5. State update: h_new = LTC(h, B_eff @ x) + W @ e
        6. Sleep: if avg_surprise > S_min, update U_target
        """
        batch_size = x.shape[0]

        # ================================================================
        # 1. Predictive Coding (Spec 2.2)
        # ================================================================
        # x_hat = C^T @ h  (C is hidden×input, so C^T @ h gives input,)
        x_pred = torch.tanh(state.h @ self.C)  # (batch, input_dim)

        # Error (innovation)
        error = x - x_pred  # (batch, input_dim)

        # ================================================================
        # 2. Surprise Gate (Spec 3.2)
        # ================================================================
        surprise, error_norm = self.compute_surprise(error, state)

        # ================================================================
        # 3. Fast Weights Update (Spec 4.2)
        # ================================================================
        # Hebbian term: outer(h, e) @ V
        # h: (batch, hidden), e: (batch, input), V: (input, rank)
        # Result: (batch, hidden, rank)

        # Efficient computation: (h @ V.T @ e.T).T won't work directly
        # Use einsum: outer[b,h,i] = h[b,h] * e[b,i]
        # Then: update[b,h,r] = sum_i outer[b,h,i] * V[i,r]
        # = sum_i h[b,h] * e[b,i] * V[i,r]
        # = h[b,h] * sum_i e[b,i] * V[i,r]
        # = h[b,h] * (e @ V)[b,r]
        # So: update = h.unsqueeze(2) * (e @ V).unsqueeze(1)

        eV = error @ self.V  # (batch, rank)
        hebbian = state.h.unsqueeze(2) * eV.unsqueeze(1)  # (batch, hidden, rank)

        # Plasticity modulation (Spec 4.2)
        # eta is (hidden,), surprise is (batch,)
        # eta * surprise: broadcast to (batch, hidden)
        plasticity = self.eta.unsqueeze(0) * surprise.unsqueeze(1)  # (batch, hidden)
        plasticity = plasticity.unsqueeze(2)  # (batch, hidden, 1)

        # Forgetting term
        forgetting = -self.forgetting_rate * (state.U - state.U_target)

        # Full update (Spec 4.2)
        dU = forgetting + plasticity * hebbian

        # Euler integration
        U_new = state.U + dU * self.dt

        # Normalize to target norm (Spec 4.2 homeostasis)
        U_norm = U_new.norm(dim=(1, 2), keepdim=True)
        scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=2.0)
        U_new = U_new * scale

        # Update state
        state.U = U_new

        # ================================================================
        # 4. Gain Modulation (Spec 2.3, 4)
        # ================================================================
        # B_eff = (1 + κ * S) * B_base
        gain = 1.0 + self.kappa * surprise.unsqueeze(1)  # (batch, 1)
        # B_base: (input, hidden), x: (batch, input)
        # x @ B_base: (batch, hidden)
        base_effect = x @ self.B_base  # (batch, hidden)
        u_eff = gain * base_effect  # (batch, hidden)

        # Add fast weights contribution: U @ V.T
        # U: (batch, hidden, rank), V: (input, rank)
        # U @ V.T: (batch, hidden, input)
        # Then @ x: (batch, hidden, input) @ (batch, input, 1) = (batch, hidden, 1)
        fast_effect = torch.bmm(state.U, self.V.T.unsqueeze(0).expand(batch_size, -1, -1))  # (batch, hidden, input)
        fast_effect = torch.bmm(fast_effect, x.unsqueeze(2)).squeeze(2)  # (batch, hidden)
        u_eff = u_eff + fast_effect * 0.1  # Scale to prevent dominance

        # ================================================================
        # 5. State Update with LTC (Spec 2.3)
        # ================================================================
        h_ltc = self.compute_ltc_update(state.h, u_eff, surprise)

        # Error injection (Spec 2.2, eq 3)
        # h_new = h_ltc + W @ e
        error_injection = error @ self.W  # (batch, hidden)

        # Combine (Spec 6 pseudocode)
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
        # 7. Sleep Consolidation (Spec 5.2)
        # ================================================================
        avg_surprise_mean = state.avg_surprise.mean()

        if avg_surprise_mean > self.S_min:
            # Consolidate U into U_target
            dU_target = self.sleep_rate * avg_surprise_mean * (state.U - state.U_target)
            state.U_target = state.U_target + dU_target

            # Homeostasis (Spec 5.2)
            U_target_norm = state.U_target.norm(dim=(1, 2), keepdim=True)
            scale = (self.target_norm / (U_target_norm + 1e-6)).clamp(max=2.0)
            state.U_target = state.U_target * scale

        return h_new, state

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        state: Optional[DREAMState] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, DREAMState]:
        """Process full sequence through cell."""
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

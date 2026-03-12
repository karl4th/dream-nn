"""
Block 5: Sleep Consolidation

Memory stabilization during high surprise:
- Multiple triggers: surprise, timer, degradation
- Consolidates fast weights into long-term memory
"""

import torch
import torch.nn as nn


class SleepConsolidation(nn.Module):
    """
    Sleep Consolidation block.

    Consolidates fast weights (U) into target weights (U_target)
    during periods of high surprise or other triggers.

    Triggers:
    1. High surprise: avg_surprise > S_min
    2. Timer: steps_since_sleep > T_sleep
    3. Degradation: error_mean > error_threshold

    Parameters
    ----------
    sleep_rate : float
        Sleep consolidation rate (ζ_sleep)
    min_surprise_for_sleep : float
        Minimum surprise threshold for sleep (S_min)
    min_steps_for_sleep : int
        Minimum steps between sleep cycles (T_sleep)
    error_threshold : float
        Error threshold for degradation-triggered sleep
    target_norm : float
        Target norm for homeostasis
    """

    def __init__(
        self,
        sleep_rate: float = 0.1,
        min_surprise_for_sleep: float = 0.5,
        min_steps_for_sleep: int = 100,
        error_threshold: float = 5.0,
        target_norm: float = 2.0
    ):
        super().__init__()
        self.sleep_rate = sleep_rate
        self.S_min = min_surprise_for_sleep
        self.T_sleep = min_steps_for_sleep
        self.error_threshold = error_threshold
        self.target_norm = target_norm

        # Step counter for timer-based trigger
        self.steps_since_sleep = 0

    def should_trigger_sleep(
        self,
        avg_surprise: torch.Tensor,
        error_mean: torch.Tensor
    ) -> bool:
        """
        Check if sleep should be triggered.

        Parameters
        ----------
        avg_surprise : torch.Tensor
            Average surprise (batch,) or scalar
        error_mean : torch.Tensor
            Error mean (batch, input_dim)

        Returns
        -------
        bool
            True if sleep should be triggered
        """
        self.steps_since_sleep += 1

        # 1. Surprise trigger
        if avg_surprise.mean().item() > self.S_min:
            return True

        # 2. Timer trigger
        if self.steps_since_sleep > self.T_sleep:
            return True

        # 3. Degradation trigger (high error)
        error_norm = error_mean.norm(dim=-1).mean().item()
        if error_norm > self.error_threshold:
            return True

        return False

    def reset_timer(self):
        """Reset sleep timer after sleep cycle."""
        self.steps_since_sleep = 0

    def forward(
        self,
        U: torch.Tensor,
        U_target: torch.Tensor,
        avg_surprise: torch.Tensor,
        error_mean: torch.Tensor,
        force: bool = False
    ) -> tuple:
        """
        Update target weights if sleep is triggered.

        Parameters
        ----------
        U : torch.Tensor
            Current fast weights (batch, hidden_dim, rank)
        U_target : torch.Tensor
            Target fast weights (batch, hidden_dim, rank)
        avg_surprise : torch.Tensor
            Average surprise (batch,)
        error_mean : torch.Tensor
            Error mean (batch, input_dim)
        force : bool
            Force sleep regardless of triggers

        Returns
        -------
        U_target_new : torch.Tensor
            Updated target weights
        triggered : bool
            Whether sleep was triggered
        """
        # Check triggers
        if not force:
            triggered = self.should_trigger_sleep(avg_surprise, error_mean)
        else:
            triggered = True

        if triggered:
            # Aggregate experience: simple moving average
            # U_agg = (1 - ζ) * U_target + ζ * U
            U_agg = (1 - self.sleep_rate) * U_target + self.sleep_rate * U

            # Consolidate with homeostasis
            U_target_new = U_agg

            # Homeostasis
            U_target_norm = U_target_new.norm(dim=(1, 2), keepdim=True)
            scale = (self.target_norm / (U_target_norm + 1e-6)).clamp(max=2.0)
            U_target_new = U_target_new * scale

            # Reset timer
            self.reset_timer()

            return U_target_new, True

        return U_target, False

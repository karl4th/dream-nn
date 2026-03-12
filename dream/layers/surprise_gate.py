"""
Block 2: Surprise Gate

Computes surprise as statistical anomaly using z-score:
- Robust statistics with clipping
- S = sigmoid((z - τ_base) / γ)
- z = (||e|| - μ) / (σ + ε)
"""

import torch
import torch.nn as nn


class SurpriseGate(nn.Module):
    """
    Surprise Gate block.

    Computes surprise as a normalized statistical anomaly,
    not just raw error magnitude.

    Parameters
    ----------
    hidden_dim : int
        Dimension of hidden state
    base_threshold : float
        Base surprise threshold (τ_base) - z-score at S=0.5
    entropy_influence : float
        Alpha (α) - entropy influence on threshold (reserved)
    surprise_temperature : float
        Gamma (γ) - surprise temperature (sigmoid steepness)
    kappa : float
        Gain modulation coefficient
    error_smoothing : float
        Beta (β) - exponential smoothing for statistics
    """

    def __init__(
        self,
        hidden_dim: int,
        base_threshold: float = 2.0,
        entropy_influence: float = 0.1,
        surprise_temperature: float = 1.0,
        kappa: float = 0.5,
        error_smoothing: float = 0.05
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = error_smoothing

        # τ_base: z-score threshold (default 2.0 = 2 std deviations)
        self.tau_base = nn.Parameter(torch.tensor(base_threshold))
        self.alpha = nn.Parameter(torch.tensor(entropy_influence))
        
        # γ: temperature (small = steep sigmoid, large = smooth)
        self.gamma = nn.Parameter(torch.tensor(surprise_temperature))
        self.kappa = nn.Parameter(torch.tensor(kappa))

    def compute_entropy(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from variance.
        H = 0.5 * log(2πe * var)
        """
        eps = 1e-6
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (variance + eps))
        return torch.clamp(entropy, 0.0, 2.0)

    def update_statistics(
        self,
        error_norm: torch.Tensor,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor
    ) -> tuple:
        """
        Update running statistics with clipping.

        Clip error to [μ, μ + 3σ] to prevent outliers from corrupting statistics.

        Parameters
        ----------
        error_norm : torch.Tensor
            Current error norm (batch,)
        mu : torch.Tensor
            Running mean of error norm
        sigma_sq : torch.Tensor
            Running variance of error norm

        Returns
        -------
        mu_new : torch.Tensor
            Updated mean
        sigma_sq_new : torch.Tensor
            Updated variance
        """
        eps = 1e-6
        sigma = torch.sqrt(sigma_sq + eps)

        # Clip error to [μ, μ + 3σ] to prevent outlier corruption
        clipped_error = torch.clamp(
            error_norm,
            min=mu,
            max=mu + 3 * sigma
        )

        # Exponential moving average update
        mu_new = (1 - self.beta) * mu + self.beta * clipped_error
        sigma_sq_new = (1 - self.beta) * sigma_sq + self.beta * (clipped_error - mu_new) ** 2

        return mu_new, sigma_sq_new

    def forward(
        self,
        error: torch.Tensor,
        error_var: torch.Tensor,
        error_mean: torch.Tensor,
        state_mu: torch.Tensor,
        state_sigma: torch.Tensor
    ) -> tuple:
        """
        Compute surprise as statistical anomaly.

        Parameters
        ----------
        error : torch.Tensor
            Prediction error (batch, input_dim)
        error_var : torch.Tensor
            Error variance (batch, input_dim) - for entropy (optional)
        error_mean : torch.Tensor
            Error mean (batch, input_dim) - for entropy (optional)
        state_mu : torch.Tensor
            Running mean of error norm (batch,) or scalar
        state_sigma : torch.Tensor
            Running std of error norm (batch,) or scalar

        Returns
        -------
        surprise : torch.Tensor
            Surprise values (batch,) in [0, 1]
        error_norm : torch.Tensor
            Error norm (batch,)
        gain : torch.Tensor
            Gain modulation (batch, 1)
        mu_new : torch.Tensor
            Updated mean
        sigma_new : torch.Tensor
            Updated std
        """
        batch_size = error.shape[0]
        eps = 1e-6

        # Error norm
        error_norm = error.norm(dim=-1)  # (batch,)

        # Update statistics with clipping
        mu_new, sigma_sq_new = self.update_statistics(
            error_norm, state_mu, state_sigma ** 2
        )
        sigma_new = torch.sqrt(sigma_sq_new + eps)

        # Entropy from variance (optional, for future use)
        variance = error_var.mean(dim=-1)
        entropy = self.compute_entropy(variance)

        # ================================================================
        # Z-score computation (KEY CHANGE)
        # ================================================================
        # z = (||e|| - μ) / (σ + ε)
        # How many standard deviations is current error from mean?
        z_score = (error_norm - mu_new) / (sigma_new + eps)

        # ================================================================
        # Surprise via sigmoid
        # ================================================================
        # S = sigmoid((z - τ_base) / γ)
        # τ_base is the z-score at which S = 0.5
        surprise = torch.sigmoid((z_score - self.tau_base) / self.gamma)

        # Gain modulation
        gain = 1.0 + self.kappa * surprise.unsqueeze(1)

        return surprise, error_norm, gain, mu_new, sigma_new

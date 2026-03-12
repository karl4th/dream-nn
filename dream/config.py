"""Configuration for DREAM cells."""

from dataclasses import dataclass


@dataclass
class DREAMConfig:
    """
    Configuration for DREAM (Dynamic Recall and Elastic Adaptive Memory) cell.

    A continuous-time RNN with modular blocks:
    - Predictive Coding (required)
    - Surprise Gate (required)
    - Fast Weights (optional)
    - Liquid Time-Constants (optional)
    - Sleep Consolidation (optional)

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : int
        Dimension of hidden state
    rank : int
        Rank of fast weights decomposition

    Block Control
    -------------
    use_fast_weights : bool
        Enable fast weights with Hebbian learning
    use_ltc : bool
        Enable Liquid Time-Constants dynamics
    use_sleep : bool
        Enable sleep consolidation

    Time Parameters
    ---------------
    time_step : float
        Integration time step (dt)

    Plasticity Parameters
    ---------------------
    forgetting_rate : float
        Forgetting rate for fast weights (λ)
    base_plasticity : float
        Base plasticity coefficient (η)

    Surprise Parameters
    -------------------
    base_threshold : float
        Base surprise threshold (τ₀)
    entropy_influence : float
        Entropy influence on threshold (α)
    surprise_temperature : float
        Surprise temperature (γ)
    kappa : float
        Gain modulation coefficient

    Smoothing Parameters
    --------------------
    error_smoothing : float
        Smoothing for error statistics (β)
    surprise_smoothing : float
        Smoothing for surprise (β_s)

    Homeostasis Parameters
    ----------------------
    target_norm : float
        Target norm for fast weights

    LTC Parameters
    --------------
    ltc_tau_sys : float
        Base system time constant
    ltc_surprise_scale : float
        Surprise modulation scaling

    Sleep Parameters
    ----------------
    sleep_rate : float
        Sleep consolidation rate
    min_surprise_for_sleep : float
        Minimum surprise for sleep activation

    Examples
    --------
    >>> config = DREAMConfig(input_dim=39, hidden_dim=256, rank=16)
    >>> cell = DREAMCell(config)
    """

    # =====================================================================
    # Model Dimensions
    # =====================================================================
    input_dim: int = 39
    hidden_dim: int = 256
    rank: int = 16

    # =====================================================================
    # Block Control - Enable/Disable individual blocks
    # =====================================================================
    use_fast_weights: bool = True
    """Enable fast weights with Hebbian learning"""

    use_ltc: bool = True
    """Enable Liquid Time-Constants dynamics"""

    use_sleep: bool = True
    """Enable sleep consolidation"""

    # =====================================================================
    # Time Parameters
    # =====================================================================
    time_step: float = 0.1
    """Integration time step (dt)"""

    # =====================================================================
    # Plasticity Parameters
    # =====================================================================
    forgetting_rate: float = 0.005
    """Forgetting rate for fast weights (λ)"""

    base_plasticity: float = 0.5
    """Base plasticity coefficient (η)"""

    # =====================================================================
    # Surprise Parameters
    # =====================================================================
    base_threshold: float = 2.0
    """Base surprise threshold (τ_base) - z-score at S=0.5"""

    entropy_influence: float = 0.1
    """Entropy influence on threshold (α) - reserved"""

    surprise_temperature: float = 1.0
    """Surprise temperature (γ) - sigmoid steepness"""

    kappa: float = 0.5
    """Gain modulation coefficient"""

    # =====================================================================
    # Smoothing Parameters
    # =====================================================================
    error_smoothing: float = 0.05
    """Smoothing for error statistics (β)"""

    surprise_smoothing: float = 0.05
    """Smoothing for surprise (β_s)"""

    # =====================================================================
    # Homeostasis Parameters
    # =====================================================================
    target_norm: float = 2.0
    """Target norm for fast weights"""

    # =====================================================================
    # LTC Parameters
    # =====================================================================
    ltc_tau_sys: float = 5.0
    """Base system time constant"""

    ltc_surprise_scale: float = 10.0
    """Surprise modulation scaling"""

    # =====================================================================
    # Sleep Parameters
    # =====================================================================
    sleep_rate: float = 0.1
    """Sleep consolidation rate (ζ)"""

    min_surprise_for_sleep: float = 0.5
    """Minimum surprise for sleep activation (S_min)"""

    min_steps_for_sleep: int = 100
    """Minimum steps between sleep cycles (T_sleep)"""

    error_threshold_for_sleep: float = 5.0
    """Error threshold for degradation-triggered sleep"""

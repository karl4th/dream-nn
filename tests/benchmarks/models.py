"""
Model wrappers for benchmark comparison.

Provides unified interface for DREAM, LSTM, and Transformer models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dream import DREAMConfig, DREAMCell, DREAMState


class DREAMReconstructor(nn.Module):
    """
    DREAM-based sequence reconstructor.

    Uses DREAM cell with persistent state for online adaptation.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        rank: int = 16,
        config: Optional[DREAMConfig] = None
    ):
        super().__init__()
        if config is None:
            config = DREAMConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                rank=rank,
                forgetting_rate=0.005,
                base_plasticity=0.5,
                base_threshold=0.3,
                entropy_influence=0.1,
                surprise_temperature=0.05,
                error_smoothing=0.05,
                surprise_smoothing=0.05,
                ltc_tau_sys=5.0,
                ltc_surprise_scale=5.0,
                kappa=0.5,
                sleep_rate=0.01,
                min_surprise_for_sleep=0.15,
            )
        self.config = config
        self.cell = DREAMCell(config)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DREAMState] = None,
        return_all: bool = True
    ) -> Tuple[torch.Tensor, DREAMState]:
        """Process sequence and reconstruct."""
        batch_size, time_steps, _ = x.shape

        if state is None:
            state = self.cell.init_state(batch_size, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(time_steps):
            x_t = x[:, t, :]
            h, state = self.cell(x_t, state)
            recon = self.decoder(h)
            if return_all:
                outputs.append(recon.unsqueeze(1))

        if return_all:
            return torch.cat(outputs, dim=1), state
        else:
            return recon, state

    def init_state(self, batch_size: int, **kwargs) -> DREAMState:
        return self.cell.init_state(batch_size, **kwargs)


class LSTMReconstructor(nn.Module):
    """
    LSTM-based sequence reconstructor.

    Match parameter count with DREAM for fair comparison.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_all: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process sequence and reconstruct."""
        batch_size = x.shape[0]

        if state is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=x.device, dtype=x.dtype)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=x.device, dtype=x.dtype)
            state = (h_0, c_0)

        output, new_state = self.lstm(x, state)
        reconstruction = self.decoder(output)

        if return_all:
            return reconstruction, new_state
        else:
            return reconstruction[:, -1, :], new_state

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM state."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                         device=device, dtype=dtype)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                         device=device, dtype=dtype)
        return (h_0, c_0)


class TransformerReconstructor(nn.Module):
    """
    Transformer-based sequence reconstructor.

    Uses positional encoding and transformer encoder.
    """

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.decoder = nn.Linear(d_model, input_dim)

        self.d_model = d_model
        self.input_dim = input_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
        return_all: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """Process sequence and reconstruct."""
        batch_size, seq_len, _ = x.shape

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        output = self.transformer(x)

        # Decode
        reconstruction = self.decoder(output)

        # State is empty for transformer (no recurrence)
        new_state = {}

        if return_all:
            return reconstruction, new_state
        else:
            return reconstruction[:, -1, :], new_state

    def init_state(self, batch_size: int, **kwargs) -> dict:
        """Initialize transformer state (empty, no recurrence)."""
        return {}


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    model_type: str,
    input_dim: int = 80,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Parameters
    ----------
    model_type : str
        One of 'dream', 'lstm', 'transformer'
    input_dim : int
        Input feature dimension
    **kwargs
        Additional arguments passed to model constructor

    Returns
    -------
    nn.Module
        Initialized model
    """
    if model_type == 'dream':
        return DREAMReconstructor(input_dim=input_dim, **kwargs)
    elif model_type == 'lstm':
        return LSTMReconstructor(input_dim=input_dim, **kwargs)
    elif model_type == 'transformer':
        return TransformerReconstructor(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

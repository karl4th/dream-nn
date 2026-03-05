"""Basic tests for DREAM cell and high-level API."""

import pytest
import torch
from dream import DREAMConfig, DREAMCell, DREAM, DREAMStack, DREAMState


class TestDREAMConfig:
    """Tests for DREAMConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DREAMConfig()
        assert config.input_dim == 39  # Default for ASR (MFCC 39D)
        assert config.hidden_dim == 256
        assert config.rank == 16
        assert config.ltc_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DREAMConfig(
            input_dim=64,
            hidden_dim=128,
            rank=8,
            ltc_enabled=False,
        )
        assert config.input_dim == 64
        assert config.hidden_dim == 128
        assert config.rank == 8
        assert config.ltc_enabled is False


class TestDREAMCell:
    """Tests for DREAMCell."""

    @pytest.fixture
    def cell(self):
        """Create test cell."""
        config = DREAMConfig(input_dim=32, hidden_dim=64, rank=8)
        return DREAMCell(config)

    def test_init(self, cell):
        """Test cell initialization."""
        assert cell.config.input_dim == 32
        assert cell.config.hidden_dim == 64
        assert cell.config.rank == 8

    def test_init_state(self, cell):
        """Test state initialization."""
        batch_size = 4
        state = cell.init_state(batch_size)
        
        assert isinstance(state, DREAMState)
        assert state.h.shape == (batch_size, cell.config.hidden_dim)
        assert state.U.shape == (batch_size, cell.config.hidden_dim, cell.config.rank)
        assert state.U_target.shape == (batch_size, cell.config.hidden_dim, cell.config.rank)

    def test_forward(self, cell):
        """Test forward pass."""
        batch_size = 4
        state = cell.init_state(batch_size)
        x = torch.randn(batch_size, cell.config.input_dim)
        
        h, new_state = cell(x, state)
        
        assert h.shape == (batch_size, cell.config.hidden_dim)
        assert isinstance(new_state, DREAMState)

    def test_forward_sequence(self, cell):
        """Test sequence processing."""
        batch_size = 4
        seq_len = 20
        state = cell.init_state(batch_size)
        x = torch.randn(batch_size, seq_len, cell.config.input_dim)
        
        output, new_state = cell.forward_sequence(x, state, return_all=True)
        
        assert output.shape == (batch_size, seq_len, cell.config.hidden_dim)
        assert isinstance(new_state, DREAMState)

    def test_per_batch_u(self, cell):
        """Test per-batch U matrices."""
        batch_size = 4
        state = cell.init_state(batch_size)
        
        # U should be per-batch
        assert state.U.shape[0] == batch_size
        
        # Different batch elements should have different U
        x = torch.randn(batch_size, cell.config.input_dim)
        _, new_state = cell(x, state)
        
        # U norms can be different per batch
        u_norms = new_state.U.norm(dim=(1, 2))
        assert u_norms.shape == (batch_size,)


class TestDREAM:
    """Tests for high-level DREAM API."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return DREAM(input_dim=32, hidden_dim=64, rank=8)

    def test_init(self, model):
        """Test model initialization."""
        assert model.hidden_dim == 64

    def test_forward(self, model):
        """Test forward pass."""
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 32)
        
        output, state = model(x)
        
        assert output.shape == (batch_size, seq_len, 64)

    def test_forward_last_only(self, model):
        """Test forward with return_sequences=False."""
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 32)
        
        output, state = model(x, return_sequences=False)
        
        assert output.shape == (batch_size, 64)

    def test_forward_with_mask(self, model):
        """Test masked forward pass."""
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 32)
        lengths = torch.tensor([15, 18, 20, 12])
        
        output, state = model.forward_with_mask(x, lengths)
        
        assert output.shape == (batch_size, seq_len, 64)


class TestDREAMStack:
    """Tests for DREAMStack."""

    @pytest.fixture
    def stack(self):
        """Create test stack."""
        return DREAMStack(
            input_dim=32,
            hidden_dims=[64, 64, 32],
            rank=8,
            dropout=0.1,
        )

    def test_init(self, stack):
        """Test stack initialization."""
        assert len(stack.layers) == 3
        assert stack.hidden_dims == [64, 64, 32]

    def test_forward(self, stack):
        """Test forward pass through stack."""
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 32)
        
        output, states = stack(x)
        
        assert output.shape == (batch_size, seq_len, 32)
        assert len(states) == 3


class TestLTC:
    """Tests for Liquid Time-Constants."""

    def test_learnable_parameters(self):
        """Test that LTC parameters are learnable."""
        config = DREAMConfig(input_dim=32, hidden_dim=64)
        cell = DREAMCell(config)

        assert isinstance(cell.tau_sys, torch.nn.Parameter)
        # tau_surprise_scale is a config parameter, not learnable
        assert hasattr(cell, 'tau_surprise_scale')

    def test_ltc_disabled(self):
        """Test LTC disabled mode."""
        config = DREAMConfig(
            input_dim=32,
            hidden_dim=64,
            ltc_enabled=False,
            ltc_tau_sys=0.0
        )
        cell = DREAMCell(config)

        assert cell.tau_sys.item() == 0.0

    def test_ltc_enabled(self):
        """Test LTC enabled mode."""
        config = DREAMConfig(input_dim=32, hidden_dim=64, ltc_enabled=True)
        cell = DREAMCell(config)

        assert cell.tau_sys.item() > 0.0


class TestStateDetachment:
    """Tests for state detachment (BPTT)."""

    def test_state_detach(self):
        """Test state detachment for truncated BPTT."""
        config = DREAMConfig(input_dim=32, hidden_dim=64)
        cell = DREAMCell(config)
        
        batch_size = 4
        state = cell.init_state(batch_size)
        x = torch.randn(batch_size, cell.config.input_dim)
        
        _, new_state = cell(x, state)
        
        # Detach state
        detached = new_state.detach()
        
        assert detached.h.grad_fn is None
        assert detached.U.grad_fn is None

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.optimizers.dld_optimizer import DLDOptimizer, OptimizationParameters
from config.base import BaseConfig

class TestDLDOptimizer:
    """Test cases for the DLD Optimizer."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return BaseConfig()
    
    @pytest.fixture
    def optimizer(self, config):
        """Create a test optimizer instance."""
        return DLDOptimizer(config.MODEL_PATH)
    
    @pytest.fixture
    def valid_params(self):
        """Create valid optimization parameters."""
        return OptimizationParameters(
            DI1=0.5,
            DI2=0.8,
            R1=0.5,
            R2=0.8,
            P_min=5.0,
            P_max=15.0,
            Gh_min=5.0,
            Gh_max=15.0,
            Gv_min=5.0,
            Gv_max=15.0,
            alpha_min=1.0,
            alpha_max=5.0,
            Q_min=0.5,
            Q_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.model is not None
    
    def test_parameter_validation_valid(self, optimizer, valid_params):
        """Test parameter validation with valid parameters."""
        errors = optimizer.validate_parameters(valid_params)
        assert len(errors) == 0
    
    def test_parameter_validation_invalid_bounds(self, optimizer):
        """Test parameter validation with invalid bounds."""
        invalid_params = OptimizationParameters(
            DI1=0.5,
            DI2=0.8,
            R1=0.5,
            R2=0.8,
            P_min=15.0,  # Invalid: min > max
            P_max=5.0,
            Gh_min=5.0,
            Gh_max=15.0,
            Gv_min=5.0,
            Gv_max=15.0,
            alpha_min=1.0,
            alpha_max=5.0,
            Q_min=0.5,
            Q_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
        
        errors = optimizer.validate_parameters(invalid_params)
        assert len(errors) > 0
        assert any("P_min must be less than P_max" in error for error in errors)
    
    def test_parameter_validation_invalid_di(self, optimizer):
        """Test parameter validation with invalid DI values."""
        invalid_params = OptimizationParameters(
            DI1=1.5,  # Invalid: > 1
            DI2=0.8,
            R1=0.5,
            R2=0.8,
            P_min=5.0,
            P_max=15.0,
            Gh_min=5.0,
            Gh_max=15.0,
            Gv_min=5.0,
            Gv_max=15.0,
            alpha_min=1.0,
            alpha_max=5.0,
            Q_min=0.5,
            Q_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
        
        errors = optimizer.validate_parameters(invalid_params)
        assert len(errors) > 0
        assert any("DI1 must be between 0 and 1" in error for error in errors)
    
    def test_optimization_small_scale(self, optimizer, valid_params):
        """Test optimization with a small number of trials."""
        # Use very small number of trials for quick testing
        valid_params.n_trials = 5
        valid_params.n_startup_trials = 2
        
        result = optimizer.optimize(valid_params)
        
        assert result is not None
        assert result.optimal_P > 0
        assert result.optimal_Gh > 0
        assert result.optimal_Gv > 0
        assert result.optimal_alpha > 0
        assert result.optimal_Q > 0
        assert result.max_separation_angle >= 0
        assert result.optimization_time > 0
        assert result.n_trials == 5 
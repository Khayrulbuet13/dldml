import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.optimizers.dld_optimizer import DLDOptimizer, OptimizationParameters
from config.base import BaseConfig

class TestDLDOptimizer:
    """Test cases for the DLD Optimizer with new 4-parameter system."""
    
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
            R1=7.5,
            R2=7.5,
            Pr_min=4.0,
            Pr_max=12.0,
            Pg_min=10.0,
            Pg_max=22.0,
            alpha_min=1.0,
            alpha_max=5.0,
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
        assert len(errors) == 0, f"Validation errors: {errors}"
    
    def test_parameter_validation_invalid_bounds(self, optimizer):
        """Test parameter validation with invalid bounds."""
        invalid_params = OptimizationParameters(
            DI1=0.5,
            DI2=0.8,
            R1=7.5,
            R2=7.5,
            Pr_min=15.0,  # Invalid: min > max
            Pr_max=5.0,
            Pg_min=10.0,
            Pg_max=22.0,
            alpha_min=1.0,
            alpha_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
        
        errors = optimizer.validate_parameters(invalid_params)
        assert len(errors) > 0
        assert any("Pr_min must be less than Pr_max" in error for error in errors)
    
    def test_parameter_validation_invalid_di(self, optimizer):
        """Test parameter validation with invalid DI values."""
        invalid_params = OptimizationParameters(
            DI1=1.5,  # Invalid: > 1
            DI2=0.8,
            R1=7.5,
            R2=7.5,
            Pr_min=4.0,
            Pr_max=12.0,
            Pg_min=10.0,
            Pg_max=22.0,
            alpha_min=1.0,
            alpha_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
        
        errors = optimizer.validate_parameters(invalid_params)
        assert len(errors) > 0
        assert any("DI1 must be between 0 and 1" in error for error in errors)
    
    def test_physical_constraint_gap_larger_than_radius(self, optimizer):
        """Test that Pg > Pr constraint is validated."""
        invalid_params = OptimizationParameters(
            DI1=0.5,
            DI2=0.8,
            R1=7.5,
            R2=7.5,
            Pr_min=4.0,
            Pr_max=12.0,
            Pg_min=5.0,  # Pg_min too small!
            Pg_max=10.0,
            alpha_min=1.0,
            alpha_max=5.0,
            n_trials=10,
            n_startup_trials=5
        )
        
        errors = optimizer.validate_parameters(invalid_params)
        assert len(errors) > 0
        assert any("larger than" in err.lower() for err in errors)
    
    def test_predict_theta_with_four_features(self, optimizer):
        """Test that prediction accepts 4 features."""
        # Should not raise exception
        try:
            theta = optimizer._predict_theta(DI=0.5, Pr=8.0, Pg=15.0, alpha=2.5)
            assert isinstance(theta, (float, np.floating))
            assert not np.isnan(theta)
        except Exception as e:
            pytest.fail(f"Prediction with 4 features failed: {e}")
    
    def test_optimization_small_scale(self, optimizer, valid_params):
        """Test optimization with a small number of trials."""
        # Use very small number of trials for quick testing
        valid_params.n_trials = 5
        valid_params.n_startup_trials = 2
        
        result = optimizer.optimize(valid_params)
        
        assert result is not None
        assert result.optimal_Pr > 0
        assert result.optimal_Pg > 0
        assert result.optimal_alpha >= 0
        assert result.max_separation_angle >= 0
        assert result.optimization_time > 0
        assert result.n_trials == 5
    
    def test_optimization_result_structure(self, optimizer, valid_params):
        """Test that optimization result has correct 3-parameter structure."""
        valid_params.n_trials = 5
        valid_params.n_startup_trials = 2
        
        result = optimizer.optimize(valid_params)
        
        # Check result has 3 optimized parameters
        assert hasattr(result, 'optimal_Pr')
        assert hasattr(result, 'optimal_Pg')
        assert hasattr(result, 'optimal_alpha')
        
        # Check no old parameters
        assert not hasattr(result, 'optimal_Gv')
        assert not hasattr(result, 'optimal_Q')
        assert not hasattr(result, 'optimal_P')
        assert not hasattr(result, 'optimal_Gh')
    
    def test_parameter_importance_keys(self, optimizer, valid_params):
        """Test that parameter importance contains correct keys."""
        valid_params.n_trials = 10
        valid_params.n_startup_trials = 3
        
        result = optimizer.optimize(valid_params)
        
        # Parameter importance should contain the 3 optimized parameters
        importance_keys = set(result.parameter_importance.keys())
        expected_keys = {'Pr', 'Pg', 'alpha'}
        
        # All expected keys should be present
        assert expected_keys.issubset(importance_keys), \
            f"Expected keys {expected_keys} not all in {importance_keys}"

import numpy as np
import optuna
import joblib
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OptimizationParameters:
    """Data class for optimization parameters."""
    # Cell Parameters
    DI1: float
    DI2: float
    R1: float
    R2: float
    # DLD Parameters
    P_min: float
    P_max: float
    Gh_min: float
    Gh_max: float
    Gv_min: float
    Gv_max: float
    alpha_min: float
    alpha_max: float
    Q_min: float
    Q_max: float
    # Optimization Settings
    n_trials: int = 100
    n_startup_trials: int = 15
    random_state: int = 42

@dataclass
class OptimizationResult:
    """Data class for optimization results."""
    optimal_P: float
    optimal_Gh: float
    optimal_Gv: float
    optimal_alpha: float
    optimal_Q: float
    max_separation_angle: float
    optimization_time: float
    n_trials: int
    best_trial_number: int
    parameter_importance: Dict[str, float]
    convergence_history: List[float]

class DLDOptimizer:
    """Main DLD optimization class using Optuna TPE sampler."""
    
    def __init__(self, model_path: Path):
        """
        Initialize the DLD optimizer.
        
        Args:
            model_path: Path to the trained ML model
        """
        self.model_path = model_path
        self.model = self._load_model()
        logger.info(f"DLD Optimizer initialized with model: {model_path}")
    
    def _load_model(self):
        """Load the trained ML model."""
        try:
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _predict_theta(self, DI: float, P: float, Gh: float, Gv: float, alpha: float, Q: float) -> float:
        """
        Predict theta using the ML model.
        
        Args:
            DI: Deformation Index
            P: Post Size
            Gh: Horizontal Gap Size
            Gv: Vertical Gap Size
            alpha: Row Shift
            Q: Flow Rate
            
        Returns:
            Predicted theta value
        """
        try:
            input_features = np.array([[DI, P, Gh, Gv, alpha, Q]])
            theta = self.model.predict(input_features)[0]
            return theta
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _create_objective_function(self, params: OptimizationParameters):
        """
        Create the objective function for Optuna optimization.
        
        Args:
            params: Optimization parameters
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Suggest parameters using Optuna's TPE sampler
            P = trial.suggest_float('P', params.P_min, params.P_max)
            Gh = trial.suggest_float('Gh', params.Gh_min, params.Gh_max)
            Gv = trial.suggest_float('Gv', params.Gv_min, params.Gv_max)
            alpha = trial.suggest_float('alpha', params.alpha_min, params.alpha_max)
            Q = trial.suggest_float('Q', params.Q_min, params.Q_max)
            
            # Constraint: Gh must be greater than P
            if Gh <= P:
                return 1e6  # Large penalty for invalid configurations
            
            # Get the separation angles from the model
            theta1 = self._predict_theta(params.DI1, P, Gh, Gv, alpha, Q)
            theta2 = self._predict_theta(params.DI2, P, Gh, Gv, alpha, Q)
            
            # Compute the objective function f(G) = |theta2 - theta1|
            f_G = abs(theta2 - theta1)
            
            # Return negative because Optuna minimizes by default
            # and we want to maximize the separation angle difference
            return -f_G
        
        return objective
    
    def optimize(self, params: OptimizationParameters) -> OptimizationResult:
        """
        Run the DLD optimization.
        
        Args:
            params: Optimization parameters
            
        Returns:
            Optimization result
        """
        logger.info("Starting DLD optimization...")
        start_time = time.time()
        
        # Create objective function
        objective = self._create_objective_function(params)
        
        # Create study with TPE sampler
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=params.n_startup_trials,
                multivariate=True,
                group=True,
                seed=params.random_state
            ),
            direction='minimize'
        )
        
        # Run optimization
        study.optimize(objective, n_trials=params.n_trials, show_progress_bar=False)
        
        # End timing
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Extract results
        best_params = study.best_params
        best_separation = -study.best_value  # Convert back to positive separation
        
        # Get parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            param_importance = {}
        
        # Get convergence history
        convergence_history = [-value for value in study.trials_dataframe()['value'].tolist()]
        
        # Create result object
        result = OptimizationResult(
            optimal_P=best_params['P'],
            optimal_Gh=best_params['Gh'],
            optimal_Gv=best_params['Gv'],
            optimal_alpha=best_params['alpha'],
            optimal_Q=best_params['Q'],
            max_separation_angle=best_separation,
            optimization_time=optimization_time,
            n_trials=len(study.trials),
            best_trial_number=study.best_trial.number,
            parameter_importance=param_importance,
            convergence_history=convergence_history
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best separation angle: {best_separation:.4f}")
        
        return result
    
    def validate_parameters(self, params: OptimizationParameters) -> List[str]:
        """
        Validate optimization parameters.
        
        Args:
            params: Optimization parameters
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check parameter bounds
        if params.P_min >= params.P_max:
            errors.append("P_min must be less than P_max")
        
        if params.Gh_min >= params.Gh_max:
            errors.append("Gh_min must be less than Gh_max")
        
        if params.Gv_min >= params.Gv_max:
            errors.append("Gv_min must be less than Gv_max")
        
        if params.alpha_min >= params.alpha_max:
            errors.append("alpha_min must be less than alpha_max")
        
        if params.Q_min >= params.Q_max:
            errors.append("Q_min must be less than Q_max")
        
        # Check DI values
        if not (0 <= params.DI1 <= 1):
            errors.append("DI1 must be between 0 and 1")
        
        if not (0 <= params.DI2 <= 1):
            errors.append("DI2 must be between 0 and 1")
        
        # Check trial numbers
        if params.n_trials < 1:
            errors.append("n_trials must be at least 1")
        
        if params.n_startup_trials < 1:
            errors.append("n_startup_trials must be at least 1")
        
        if params.n_startup_trials > params.n_trials:
            errors.append("n_startup_trials cannot be greater than n_trials")
        
        return errors 
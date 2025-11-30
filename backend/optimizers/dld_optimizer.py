import numpy as np
import optuna
import joblib
import time
import logging
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .optimization_model import OptimizationModel

logger = logging.getLogger(__name__)

@dataclass
class OptimizationParameters:
    """Data class for optimization parameters."""
    # Cell Parameters
    DI1: float          # Deformation Index cell 1
    DI2: float          # Deformation Index cell 2
    R1: float           # Cell radius 1 (informational)
    R2: float           # Cell radius 2 (informational)
    
    # DLD Geometry Parameters (3 continuous parameters to optimize)
    Pr_min: float       # Pillar radius min (μm)
    Pr_max: float       # Pillar radius max (μm)
    Pg_min: float       # Pillar gap min (μm)
    Pg_max: float       # Pillar gap max (μm)
    alpha_min: float    # Row shift angle min (degrees)
    alpha_max: float    # Row shift angle max (degrees)
    
    # Optimization Settings
    n_trials: int = 100
    n_startup_trials: int = 15
    random_state: int = 42

@dataclass
class OptimizationResult:
    """Data class for optimization results."""
    optimal_Pr: float      # Optimal pillar radius
    optimal_Pg: float      # Optimal pillar gap
    optimal_alpha: float   # Optimal row shift angle
    max_separation_angle: float
    optimization_time: float
    n_trials: int
    best_trial_number: int
    parameter_importance: Dict[str, float]
    convergence_history: List[float]
    study_data: Dict[str, Any]  # Study data for advanced visualizations
    trials_dataframe: Dict[str, Any]  # Trials dataframe for plotting

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
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating a simple fallback model...")
                return self._create_fallback_model()
            
            # Try to load model directly (joblib handles class resolution)
            try:
                model = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
                
                # Verify it's a valid model
                if hasattr(model, 'xgb_model') and hasattr(model, 'scaler'):
                    logger.info("✓ Loaded OptimizationModel with XGBoost")
                    logger.info(f"✓ Feature names: {model.feature_names}")
                    logger.info(f"✓ Has selector: {model.selector is not None}")
                    return model
                else:
                    logger.warning("Model missing required attributes, using fallback")
                    return self._create_fallback_model()
                    
            except Exception as load_error:
                logger.error(f"Model loading failed: {load_error}")
                raise load_error
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Creating a simple fallback model...")
            return self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model when the trained model is not available."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create a simple model with default parameters
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Train on dummy data: 4 features [DI, Pr, Pg, alpha]
        import numpy as np
        np.random.seed(42)
        X = np.random.rand(100, 4)  # 4 features
        y = np.random.rand(100) * 10  # Random theta values
        
        model.fit(X, y)
        logger.info("Fallback model created with 4 features: [DI, Pr, Pg, alpha]")
        return model
    
    def _predict_theta(self, DI: float, Pr: float, Pg: float, alpha: float) -> float:
        """
        Predict migration angle theta using the ML model.
        
        Args:
            DI: Deformation Index [0-1]
            Pr: Pillar Radius (μm)
            Pg: Pillar Gap (μm)
            alpha: Row Shift Angle (degrees, continuous representation of periodicity)
            
        Returns:
            Predicted migration angle theta (degrees)
        """
        try:
            # Model expects 4 features: [DI, Pr, Pg, alpha]
            input_features = np.array([[DI, Pr, Pg, alpha]])
            
            # Use the model's predict method (works for both OptimizationModel and fallback)
            theta = self.model.predict(input_features)[0]
            
            return theta
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(f"Input was: DI={DI}, Pr={Pr}, Pg={Pg}, alpha={alpha}")
            raise
    
    def _create_objective_function(self, params: OptimizationParameters):
        """
        Create the objective function for Optuna optimization.
        
        Optimizes 3 continuous parameters: Pr, Pg, alpha
        Objective: maximize |theta(DI2) - theta(DI1)|
        
        Args:
            params: Optimization parameters
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Suggest parameters using Optuna's TPE sampler
            Pr = trial.suggest_float('Pr', params.Pr_min, params.Pr_max)
            Pg = trial.suggest_float('Pg', params.Pg_min, params.Pg_max)
            alpha = trial.suggest_float('alpha', params.alpha_min, params.alpha_max)
            
            # Physical constraint: Gap must be larger than pillar radius
            # to prevent clogging and maintain fluid flow
            d_safety = 0.5  # Safety margin in micrometers
            if Pg <= Pr + d_safety:
                return 1e6  # Large penalty for invalid configurations
            
            # Predict migration angles for both cell types
            theta1 = self._predict_theta(params.DI1, Pr, Pg, alpha)
            theta2 = self._predict_theta(params.DI2, Pr, Pg, alpha)
            
            # Objective function: maximize separation angle difference
            # f(G) = |theta2 - theta1|
            separation_angle = abs(theta2 - theta1)
            
            # Return negative because Optuna minimizes by default
            return -separation_angle
        
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
        # Use ML model's feature importance directly (more reliable than Optuna importance)
        try:
            if hasattr(self.model, 'xgb_model') and hasattr(self.model, 'feature_names'):
                # Extract XGBoost feature importance
                xgb_importance = self.model.xgb_model.get_score(importance_type='gain')
                feature_names = self.model.feature_names
                
                logger.info(f"XGBoost raw importance: {xgb_importance}")
                logger.info(f"Feature names: {feature_names}")
                
                # Build feature importance dict
                all_importance = {}
                for feat_key, score in xgb_importance.items():
                    feat_idx = int(feat_key.replace('f', ''))
                    if feat_idx < len(feature_names):
                        # Clean numpy string if needed
                        feat_name = str(feature_names[feat_idx])
                        if 'np.str_' in feat_name:
                            import re
                            match = re.search(r"'([^']+)'", feat_name)
                            feat_name = match.group(1) if match else feat_name
                        all_importance[feat_name.strip()] = float(score)
                
                logger.info(f"Mapped importance: {all_importance}")
                
                # Extract only optimization parameters (exclude DI)
                param_importance = {k: v for k, v in all_importance.items() 
                                   if k in ['Pr', 'Pg', 'alpha']}
                
                # If any parameter missing, use small value
                for param in ['Pr', 'Pg', 'alpha']:
                    if param not in param_importance:
                        param_importance[param] = 0.01
                
                # Normalize
                total = sum(param_importance.values())
                if total > 0:
                    param_importance = {k: v/total for k, v in param_importance.items()}
                
                logger.info(f"Final parameter importance: {param_importance}")
            else:
                # Fallback
                logger.warning("Model structure doesn't support importance extraction")
                param_importance = {'Pr': 0.33, 'Pg': 0.33, 'alpha': 0.34}
        except Exception as e:
            logger.error(f"Parameter importance calculation failed: {e}", exc_info=True)
            param_importance = {'Pr': 0.33, 'Pg': 0.33, 'alpha': 0.34}
        
        # Get convergence history
        convergence_history = [-value for value in study.trials_dataframe()['value'].tolist()]
        
        # Prepare study data for advanced visualizations
        study_data = {
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'direction': study.direction,
            'sampler_name': study.sampler.__class__.__name__
        }
        
        # Prepare trials dataframe for plotting
        try:
            trials_df = study.trials_dataframe()
            # Convert to dict for JSON serialization
            trials_dataframe = {
                'columns': trials_df.columns.tolist(),
                'data': trials_df.values.tolist(),
                'index': trials_df.index.tolist()
            }
        except Exception as e:
            logger.warning(f"Could not prepare trials dataframe: {e}")
            trials_dataframe = {'columns': [], 'data': [], 'index': []}
        
        # Create result object
        result = OptimizationResult(
            optimal_Pr=best_params['Pr'],
            optimal_Pg=best_params['Pg'],
            optimal_alpha=best_params['alpha'],
            max_separation_angle=best_separation,
            optimization_time=optimization_time,
            n_trials=len(study.trials),
            best_trial_number=study.best_trial.number,
            parameter_importance=param_importance,
            convergence_history=convergence_history,
            study_data=study_data,
            trials_dataframe=trials_dataframe
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
        if params.Pr_min >= params.Pr_max:
            errors.append("Pr_min must be less than Pr_max")
        
        if params.Pg_min >= params.Pg_max:
            errors.append("Pg_min must be less than Pg_max")
        
        if params.alpha_min >= params.alpha_max:
            errors.append("alpha_min must be less than alpha_max")
        
        # Check DI values are in valid range
        if not (0 <= params.DI1 <= 1):
            errors.append("DI1 must be between 0 and 1")
        
        if not (0 <= params.DI2 <= 1):
            errors.append("DI2 must be between 0 and 1")
        
        # Check physical constraint: gap must be larger than pillar radius
        if params.Pg_min <= params.Pr_max:
            errors.append("Minimum gap (Pg_min) must be larger than maximum pillar radius (Pr_max)")
        
        # Check positive values
        if params.Pr_min <= 0:
            errors.append("Pr_min must be positive")
        
        if params.Pg_min <= 0:
            errors.append("Pg_min must be positive")
        
        if params.alpha_min < 0:
            errors.append("alpha_min must be non-negative")
        
        # Check trial numbers
        if params.n_trials < 1:
            errors.append("n_trials must be at least 1")
        
        if params.n_startup_trials < 1:
            errors.append("n_startup_trials must be at least 1")
        
        if params.n_startup_trials > params.n_trials:
            errors.append("n_startup_trials cannot be greater than n_trials")
        
        return errors 
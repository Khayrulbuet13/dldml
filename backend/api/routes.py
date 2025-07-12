from fastapi import APIRouter, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import logging

from .models import (
    OptimizationRequest, 
    OptimizationResponse, 
    ErrorResponse, 
    HealthResponse
)
from ..optimizers.dld_optimizer import DLDOptimizer, OptimizationParameters
from config.base import BaseConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["DLD Optimization"])

# Global optimizer instance
_optimizer: DLDOptimizer = None

def get_optimizer() -> DLDOptimizer:
    """Dependency to get the optimizer instance."""
    global _optimizer
    if _optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    return _optimizer

def initialize_optimizer(config: BaseConfig) -> DLDOptimizer:
    """Initialize the optimizer with the given configuration."""
    global _optimizer
    try:
        _optimizer = DLDOptimizer(config.MODEL_PATH)
        logger.info("Optimizer initialized successfully")
        return _optimizer
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        raise

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _optimizer
    return HealthResponse(
        status="healthy" if _optimizer is not None else "unhealthy",
        version="1.0.0",
        model_loaded=_optimizer is not None
    )

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_dld(
    request: OptimizationRequest,
    optimizer: DLDOptimizer = Depends(get_optimizer)
):
    """
    Optimize DLD geometry parameters.
    
    This endpoint takes optimization parameters and returns the optimal DLD geometry
    configuration that maximizes the separation angle difference between two cells.
    """
    try:
        # Convert request to optimization parameters
        params = OptimizationParameters(
            DI1=request.DI1,
            DI2=request.DI2,
            R1=request.R1,
            R2=request.R2,
            P_min=request.P_min,
            P_max=request.P_max,
            Gh_min=request.Gh_min,
            Gh_max=request.Gh_max,
            Gv_min=request.Gv_min,
            Gv_max=request.Gv_max,
            alpha_min=request.alpha_min,
            alpha_max=request.alpha_max,
            Q_min=request.Q_min,
            Q_max=request.Q_max,
            n_trials=request.n_trials,
            n_startup_trials=request.n_startup_trials,
            random_state=request.random_state
        )
        
        # Validate parameters
        validation_errors = optimizer.validate_parameters(params)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Validation errors: {', '.join(validation_errors)}"
            )
        
        # Run optimization
        result = optimizer.optimize(params)
        
        # Convert result to response model
        response = OptimizationResponse(
            optimal_P=result.optimal_P,
            optimal_Gh=result.optimal_Gh,
            optimal_Gv=result.optimal_Gv,
            optimal_alpha=result.optimal_alpha,
            optimal_Q=result.optimal_Q,
            max_separation_angle=result.max_separation_angle,
            optimization_time=result.optimization_time,
            n_trials=result.n_trials,
            best_trial_number=result.best_trial_number,
            parameter_importance=result.parameter_importance,
            convergence_history=result.convergence_history,
            study_data=result.study_data,
            trials_dataframe=result.trials_dataframe
        )
        
        logger.info(f"Optimization completed successfully in {result.optimization_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/parameters", response_model=Dict[str, Any])
async def get_default_parameters():
    """Get default optimization parameters."""
    return {
        "DI1": 0.5,
        "DI2": 0.8,
        "R1": 0.5,
        "R2": 0.8,
        "P_min": 5.0,
        "P_max": 15.0,
        "Gh_min": 5.0,
        "Gh_max": 15.0,
        "Gv_min": 5.0,
        "Gv_max": 15.0,
        "alpha_min": 1.0,
        "alpha_max": 5.0,
        "Q_min": 0.5,
        "Q_max": 5.0,
        "n_trials": 100,
        "n_startup_trials": 15,
        "random_state": 42
    }

 
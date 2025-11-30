from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime

class OptimizationRequest(BaseModel):
    """Request model for optimization API."""
    
    # Cell Parameters
    DI1: float = Field(..., ge=0.0, le=1.0, description="Deformation Index of Cell 1")
    DI2: float = Field(..., ge=0.0, le=1.0, description="Deformation Index of Cell 2")
    R1: float = Field(..., ge=0.0, description="Radius of Cell 1 (μm)")
    R2: float = Field(..., ge=0.0, description="Radius of Cell 2 (μm)")
    
    # DLD Geometry Parameters
    Pr_min: float = Field(..., gt=0.0, description="Minimum Pillar Radius (μm)")
    Pr_max: float = Field(..., gt=0.0, description="Maximum Pillar Radius (μm)")
    Pg_min: float = Field(..., gt=0.0, description="Minimum Pillar Gap (μm)")
    Pg_max: float = Field(..., gt=0.0, description="Maximum Pillar Gap (μm)")
    alpha_min: float = Field(..., ge=0.0, description="Minimum Row Shift Angle (degrees)")
    alpha_max: float = Field(..., ge=0.0, description="Maximum Row Shift Angle (degrees)")
    
    # Optimization Settings
    n_trials: int = Field(100, ge=1, le=1000, description="Number of optimization trials")
    n_startup_trials: int = Field(15, ge=1, le=100, description="Number of startup trials")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    @validator('Pr_max')
    def validate_Pr_max(cls, v, values):
        if 'Pr_min' in values and v <= values['Pr_min']:
            raise ValueError('Pr_max must be greater than Pr_min')
        return v
    
    @validator('Pg_max')
    def validate_Pg_max(cls, v, values):
        if 'Pg_min' in values and v <= values['Pg_min']:
            raise ValueError('Pg_max must be greater than Pg_min')
        return v
    
    @validator('alpha_max')
    def validate_alpha_max(cls, v, values):
        if 'alpha_min' in values and v <= values['alpha_min']:
            raise ValueError('alpha_max must be greater than alpha_min')
        return v
    
    @validator('n_startup_trials')
    def validate_n_startup_trials(cls, v, values):
        if 'n_trials' in values and v > values['n_trials']:
            raise ValueError('n_startup_trials cannot be greater than n_trials')
        return v

class OptimizationResponse(BaseModel):
    """Response model for optimization API."""
    
    # Optimal Parameters
    optimal_Pr: float = Field(..., description="Optimal Pillar Radius (μm)")
    optimal_Pg: float = Field(..., description="Optimal Pillar Gap (μm)")
    optimal_alpha: float = Field(..., description="Optimal Row Shift Angle (degrees)")
    
    # Results
    max_separation_angle: float = Field(..., description="Maximum separation angle difference (degrees)")
    optimization_time: float = Field(..., description="Optimization time in seconds")
    n_trials: int = Field(..., description="Number of trials completed")
    best_trial_number: int = Field(..., description="Best trial number")
    
    # Additional Information
    parameter_importance: Dict[str, float] = Field(..., description="Parameter importance scores")
    convergence_history: List[float] = Field(..., description="Convergence history")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Optimization timestamp")
    status: str = Field("success", description="Optimization status")
    
    # Advanced visualization data
    study_data: Dict[str, Any] = Field(..., description="Study data for advanced visualizations")
    trials_dataframe: Dict[str, Any] = Field(..., description="Trials dataframe for plotting")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    
    class Config:
        protected_namespaces = () 
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime

class OptimizationRequest(BaseModel):
    """Request model for optimization API."""
    
    # Cell Parameters
    DI1: float = Field(..., ge=0.0, le=1.0, description="Deformation Index of Cell 1")
    DI2: float = Field(..., ge=0.0, le=1.0, description="Deformation Index of Cell 2")
    R1: float = Field(..., ge=0.0, le=1.0, description="Radius of Cell 1")
    R2: float = Field(..., ge=0.0, le=1.0, description="Radius of Cell 2")
    
    # DLD Parameters
    P_min: float = Field(..., gt=0.0, description="Minimum Post Size (μm)")
    P_max: float = Field(..., gt=0.0, description="Maximum Post Size (μm)")
    Gh_min: float = Field(..., gt=0.0, description="Minimum Horizontal Gap Size (μm)")
    Gh_max: float = Field(..., gt=0.0, description="Maximum Horizontal Gap Size (μm)")
    Gv_min: float = Field(..., gt=0.0, description="Minimum Vertical Gap Size (μm)")
    Gv_max: float = Field(..., gt=0.0, description="Maximum Vertical Gap Size (μm)")
    alpha_min: float = Field(..., gt=0.0, description="Minimum Row Shift (degrees)")
    alpha_max: float = Field(..., gt=0.0, description="Maximum Row Shift (degrees)")
    Q_min: float = Field(..., gt=0.0, description="Minimum Flow Rate (μL/min)")
    Q_max: float = Field(..., gt=0.0, description="Maximum Flow Rate (μL/min)")
    
    # Optimization Settings
    n_trials: int = Field(100, ge=1, le=1000, description="Number of optimization trials")
    n_startup_trials: int = Field(15, ge=1, le=100, description="Number of startup trials")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    @validator('P_max')
    def validate_P_max(cls, v, values):
        if 'P_min' in values and v <= values['P_min']:
            raise ValueError('P_max must be greater than P_min')
        return v
    
    @validator('Gh_max')
    def validate_Gh_max(cls, v, values):
        if 'Gh_min' in values and v <= values['Gh_min']:
            raise ValueError('Gh_max must be greater than Gh_min')
        return v
    
    @validator('Gv_max')
    def validate_Gv_max(cls, v, values):
        if 'Gv_min' in values and v <= values['Gv_min']:
            raise ValueError('Gv_max must be greater than Gv_min')
        return v
    
    @validator('alpha_max')
    def validate_alpha_max(cls, v, values):
        if 'alpha_min' in values and v <= values['alpha_min']:
            raise ValueError('alpha_max must be greater than alpha_min')
        return v
    
    @validator('Q_max')
    def validate_Q_max(cls, v, values):
        if 'Q_min' in values and v <= values['Q_min']:
            raise ValueError('Q_max must be greater than Q_min')
        return v
    
    @validator('n_startup_trials')
    def validate_n_startup_trials(cls, v, values):
        if 'n_trials' in values and v > values['n_trials']:
            raise ValueError('n_startup_trials cannot be greater than n_trials')
        return v

class OptimizationResponse(BaseModel):
    """Response model for optimization API."""
    
    # Optimal Parameters
    optimal_P: float = Field(..., description="Optimal Post Size (μm)")
    optimal_Gh: float = Field(..., description="Optimal Horizontal Gap Size (μm)")
    optimal_Gv: float = Field(..., description="Optimal Vertical Gap Size (μm)")
    optimal_alpha: float = Field(..., description="Optimal Row Shift (degrees)")
    optimal_Q: float = Field(..., description="Optimal Flow Rate (μL/min)")
    
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
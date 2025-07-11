import os
from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Production configuration."""
    
    DEBUG = False
    LOG_LEVEL = "WARNING"
    
    # Production-specific settings
    RELOAD = False
    AUTO_RELOAD = False
    
    # Production database
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dld_optimization_prod.db")
    
    # Security settings for production
    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")
    
    # Production CORS settings
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").split(",") 
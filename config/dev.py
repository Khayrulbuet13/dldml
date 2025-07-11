from .base import BaseConfig

class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    
    # Development-specific settings
    RELOAD = True
    AUTO_RELOAD = True
    
    # Development database
    DATABASE_URL = "sqlite:///./dld_optimization_dev.db" 
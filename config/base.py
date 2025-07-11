import os
from pathlib import Path

class BaseConfig:
    """Base configuration class with common settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    APP_DIR = PROJECT_ROOT / "app"
    BACKEND_DIR = PROJECT_ROOT / "backend"
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Model settings
    MODEL_FILENAME = "trained_model.joblib"
    MODEL_PATH = MODELS_DIR / MODEL_FILENAME
    
    # Optimization settings
    DEFAULT_N_TRIALS = 100
    DEFAULT_N_STARTUP_TRIALS = 15
    DEFAULT_RANDOM_STATE = 42
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4
    
    # Streamlit settings
    STREAMLIT_PORT = 8501
    STREAMLIT_HOST = "0.0.0.0"
    
    # Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
    
    # Database (for future use)
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dld_optimization.db")
    
    # CORS settings
    CORS_ORIGINS = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ] 
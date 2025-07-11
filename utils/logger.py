import logging
import sys
from pathlib import Path
from typing import Optional
from config.base import BaseConfig

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_logger(name: str, config: BaseConfig = None) -> logging.Logger:
    """
    Get a logger configured for the project.
    
    Args:
        name: Logger name
        config: Configuration object (optional)
    
    Returns:
        Configured logger instance
    """
    
    if config is None:
        config = BaseConfig()
    
    log_file = config.LOGS_DIR / f"{name}.log"
    
    return setup_logger(
        name=name,
        log_file=log_file,
        level=config.LOG_LEVEL,
        format_string=config.LOG_FORMAT
    ) 
#!/usr/bin/env python3
"""
Script to start the DLD Optimization backend API service.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from config.base import BaseConfig
from backend.api.app import create_app

def main():
    """Start the backend API service."""
    
    # Load configuration
    config = BaseConfig()
    
    # Create FastAPI app
    app = create_app(config)
    
    # Start server
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        workers=config.API_WORKERS if not config.DEBUG else 1,
        log_level=config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main() 
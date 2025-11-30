#!/usr/bin/env python3
"""
Main entry point for the DLD Optimization Project.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.base import BaseConfig
from backend.api.app import create_app
from app.main import main as streamlit_main

def start_backend():
    """Start the backend API service."""
    import uvicorn
    from config.base import BaseConfig
    
    config = BaseConfig()
    app = create_app(config)
    
    print(f"Starting backend API on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        workers=config.API_WORKERS if not config.DEBUG else 1
    )

def start_frontend():
    """Start the Streamlit frontend."""
    print("Starting Streamlit frontend...")
    streamlit_main()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DLD Optimization Project")
    parser.add_argument(
        "service",
        choices=["backend", "frontend", "both"],
        help="Service to start"
    )
    
    args = parser.parse_args()
    
    if args.service == "backend":
        start_backend()
    elif args.service == "frontend":
        start_frontend()
    elif args.service == "both":
        print("Starting both backend and frontend...")
        print("Note: You may want to run them in separate terminals for better control.")
        start_backend()

if __name__ == "__main__":
    main() 
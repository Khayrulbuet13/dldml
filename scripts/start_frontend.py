#!/usr/bin/env python3
"""
Script to start the DLD Optimization Streamlit frontend.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
from config.base import BaseConfig

def main():
    """Start the Streamlit frontend."""
    
    # Load configuration
    config = BaseConfig()
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(config.STREAMLIT_PORT)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = config.STREAMLIT_HOST
    
    # Start Streamlit
    subprocess.run([
        "streamlit", "run", 
        str(project_root / "app" / "main.py"),
        "--server.port", str(config.STREAMLIT_PORT),
        "--server.address", config.STREAMLIT_HOST
    ])

if __name__ == "__main__":
    main() 
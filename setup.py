#!/usr/bin/env python3
"""
Setup script for the DLD Optimization Project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = requirements_path.read_text().splitlines() if requirements_path.exists() else []

setup(
    name="dld-optimization",
    version="1.0.0",
    description="A full-stack web application for optimizing DLD geometry parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DLD Optimization Team",
    author_email="team@example.com",
    url="https://github.com/your-org/dld-optimization",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="dld optimization machine learning streamlit fastapi",
    project_urls={
        "Bug Reports": "https://github.com/your-org/dld-optimization/issues",
        "Source": "https://github.com/your-org/dld-optimization",
        "Documentation": "https://github.com/your-org/dld-optimization/docs",
    },
) 
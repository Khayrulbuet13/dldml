# DLD Optimization Project

A comprehensive tool for optimizing Deterministic Lateral Displacement (DLD) geometry parameters using machine learning and optimization algorithms.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose (for Option 1)
- Python 3.10+ and pip (for Option 2)

## 🏃‍♂️ Running the Application

The application consists of two components:
- **Backend API** (FastAPI) - Handles optimization requests
- **Frontend** (Streamlit) - User interface

### Option 1: Docker Deployment (Recommended)

The easiest way to run the application is using Docker Compose, which handles all dependencies and networking automatically.

1. **Start all services with Docker Compose**
   ```bash
   # Start backend and frontend
   docker-compose up -d
   
   # Or start with nginx proxy (production mode)
   docker-compose --profile production up -d
   ```

2. **Access the Application**
   - **Frontend (Main App)**: http://localhost:8501
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **With Nginx**: http://localhost (production mode)

3. **Stop the services**
   ```bash
   docker-compose down
   ```

4. **View logs**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

### Option 2: Using the Run Script

For local development with automatic service management:

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Khayrulbuet13/dldml
   cd dldml
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using venv
   python -m venv dld_env
   source dld_env/bin/activate  # On Linux/Mac
   # or
   dld_env\Scripts\activate     # On Windows
   
   # Or using virtualenvwrapper
   mkvirtualenv dld
   workon dld
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

#### Running the Application

```bash
# Make the script executable
chmod +x run.sh

# Run the application
./run.sh
```

This script will:
- Activate the virtual environment
- Stop any existing services
- Start the backend API
- Start the Streamlit frontend
- Handle graceful shutdown on exit

**Access the Application**:
- **Frontend (Main App)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Option 3: Web Implementation

Access the application directly through the web interface:

**🌐 Live Application**: https://dldml.khayrul.me/

No installation or setup required - just open the link in your browser and start optimizing DLD parameters immediately.



## 📊 Using the Application

### 1. Access the Web Interface
Open your browser and go to: **http://localhost:8501**

### 2. Input Parameters
Use the sidebar to configure:
- **Cell Parameters**: DI1, DI2, R1, R2 (deformation indices and radii)
- **DLD Parameters**: P, Gh, Gv, alpha, Q (geometry parameters)
- **Optimization Settings**: Number of trials, random state

### 3. Run Optimization
1. Click the "🚀 Run Optimization" button
2. Wait for the optimization to complete
3. View results in the main panel

### 4. View Results
The application displays:
- **Optimal Parameters**: Best found values
- **Maximum Separation Angle**: Objective function value
- **Parameter Importance**: Which parameters matter most
- **Convergence History**: Optimization progress
- **Visualizations**: Charts and graphs

## 🔌 API Endpoints

### Health Check
```bash
GET http://localhost:8000/api/v1/health
```

### Get Default Parameters
```bash
GET http://localhost:8000/api/v1/parameters
```

### Run Optimization
```bash
POST http://localhost:8000/api/v1/optimize
Content-Type: application/json

{
  "DI1": 0.5,
  "DI2": 0.8,
  "R1": 0.5,
  "R2": 0.8,
  "P_min": 5.0,
  "P_max": 15.0,
  "Gh_min": 5.0,
  "Gh_max": 15.0,
  "Gv_min": 5.0,
  "Gv_max": 15.0,
  "alpha_min": 1.0,
  "alpha_max": 5.0,
  "Q_min": 0.5,
  "Q_max": 5.0,
  "n_trials": 100,
  "n_startup_trials": 15,
  "random_state": 42
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Backend API is not available" Error
**Problem**: Streamlit can't connect to the backend
**Solution**: 
- Ensure the backend is running on port 8000
- Check that both services are using the same virtual environment
- Verify no firewall blocking the connection
- For Docker: Check if containers are running with `docker-compose ps`

#### 2. Docker Issues
**Problem**: Containers fail to start
**Solution**:
```bash
# Check container logs
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check container status
docker-compose ps
```

#### 3. "No module named 'numpy._core'" Error
**Problem**: Version mismatch between numpy versions
**Solution**:
```bash
pip install "numpy>=1.24.0,<2.0"
```

#### 4. Model Loading Issues
**Problem**: Scikit-learn version mismatch
**Solution**:
```bash
pip install --upgrade scikit-learn
```

#### 5. Port Already in Use
**Problem**: Port 8000 or 8501 is occupied
**Solution**:
```bash
# Kill process using the port
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9

# Or use different ports
uvicorn backend.api.app:app --port 8001
streamlit run app/main.py --server.port 8502
```

### Debug Mode
To run with debug information:
```bash
# Backend with debug
uvicorn backend.api.app:app --reload --log-level debug

# Streamlit with debug
streamlit run app/main.py --logger.level debug

# Docker with debug
docker-compose up --build
```

## 📁 Project Structure

```
dldml/
├── app/                          # Streamlit frontend
│   └── main.py                   # Main Streamlit application
├── backend/                      # FastAPI backend
│   ├── api/                      # API routes and models
│   │   ├── app.py                # FastAPI application
│   │   ├── models.py             # Data models
│   │   └── routes.py             # API endpoints
│   └── optimizers/               # Optimization algorithms
│       ├── dld_optimizer.py      # DLD optimization logic
│       └── optimization_model.py # Optimization models
├── config/                       # Configuration files
├── models/                       # Trained ML models
├── data/                         # Data files
├── tests/                        # Test files
├── scripts/                      # Utility scripts
├── utils/                        # Utility functions
├── logs/                         # Application logs
├── Dockerfile.backend            # Backend Docker configuration
├── Dockerfile.frontend           # Frontend Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── nginx.conf                    # Nginx reverse proxy configuration
├── run.sh                        # Local development script
├── start_backend.py              # Backend startup script
├── main.py                       # Main application entry point
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```   

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📈 Performance

**Default Settings**:
- Trials: 100
- Startup trials: 15
- Timeout: 5 minutes
- Memory usage: ~500MB

For better results, increase the number of trials in the optimization settings.



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs in the terminal or Docker containers
3. Check the API documentation at http://localhost:8000/docs
4. Open an issue on GitHub

## 🔄 Updates

To update the application:
```bash
git pull origin main
pip install -r requirements.txt --upgrade

# For Docker deployment
docker-compose down
docker-compose pull
docker-compose up -d
```

---

**Happy Optimizing! 🚀** 
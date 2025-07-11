# DLD Optimization Project

A comprehensive tool for optimizing Deterministic Lateral Displacement (DLD) geometry parameters using machine learning and optimization algorithms.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dld_optimization_project
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

## 🏃‍♂️ Running the Application

The application consists of two components:
- **Backend API** (FastAPI) - Handles optimization requests
- **Frontend** (Streamlit) - User interface

### Option 1: Run Both Services (Recommended)

1. **Start the Backend API**
   ```bash
   # Activate virtual environment first
   source ~/.virtualenvs/dld/bin/activate  # or your virtual env path
   
   # Start the backend
   uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Frontend** (in a new terminal)
   ```bash
   # Activate virtual environment
   source ~/.virtualenvs/dld/bin/activate
   
   # Start Streamlit
   streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
   ```

3. **Access the Application**
   - **Frontend (Main App)**: http://localhost:8501
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

### Option 2: Using the Main Script

```bash
# Start backend only
python main.py backend

# Start frontend only
python main.py frontend

# Start both (backend will start first)
python main.py both
```

## 🔧 Configuration

### Environment Variables
The application uses the following environment variables (optional):

```bash
export SECRET_KEY="your-secret-key-here"
export DATABASE_URL="sqlite:///./dld_optimization.db"
```

### Configuration Files
- `config/base.py` - Base configuration
- `config/dev.py` - Development settings
- `config/prod.py` - Production settings

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

#### 2. "No module named 'numpy._core'" Error
**Problem**: Version mismatch between numpy versions
**Solution**:
```bash
pip install "numpy>=1.24.0,<2.0"
```

#### 3. Model Loading Issues
**Problem**: Scikit-learn version mismatch
**Solution**:
```bash
pip install --upgrade scikit-learn
```

#### 4. Port Already in Use
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
```

## 📁 Project Structure

```
dld_optimization_project/
├── app/                    # Streamlit frontend
│   └── main.py
├── backend/               # FastAPI backend
│   ├── api/              # API routes and models
│   ├── optimizers/       # Optimization algorithms
│   └── services/         # Business logic
├── config/               # Configuration files
├── models/               # Trained ML models
├── data/                 # Data files
├── tests/                # Test files
├── scripts/              # Utility scripts
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📈 Performance

### Optimization Parameters
- **Default trials**: 100
- **Startup trials**: 15
- **Timeout**: 5 minutes
- **Memory usage**: ~500MB

### Scaling
For larger optimizations:
- Increase `n_trials` for better results
- Adjust parameter ranges based on domain knowledge
- Consider running multiple instances for parallel optimization

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
2. Review the logs in the terminal
3. Check the API documentation at http://localhost:8000/docs
4. Open an issue on GitHub

## 🔄 Updates

To update the application:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**Happy Optimizing! 🚀** 
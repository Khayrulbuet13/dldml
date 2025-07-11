import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.base import BaseConfig

# Configuration
config = BaseConfig()
API_BASE_URL = f"http://{config.API_HOST}:{config.API_PORT}/api/v1"

# Page configuration
st.set_page_config(
    page_title="DLD Optimization Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json()["status"] == "healthy"
    except:
        return False

def get_default_parameters() -> Dict[str, Any]:
    """Get default parameters from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/parameters", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback defaults
    return {
        "DI1": 0.5, "DI2": 0.8, "R1": 0.5, "R2": 0.8,
        "P_min": 5.0, "P_max": 15.0,
        "Gh_min": 5.0, "Gh_max": 15.0,
        "Gv_min": 5.0, "Gv_max": 15.0,
        "alpha_min": 1.0, "alpha_max": 5.0,
        "Q_min": 0.5, "Q_max": 5.0,
        "n_trials": 100, "n_startup_trials": 15, "random_state": 42
    }

def run_optimization(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run optimization using the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/optimize",
            json=parameters,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Optimization timed out. Please try with fewer trials.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the optimization API. Please ensure the backend is running.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def create_parameter_importance_chart(importance_data: Dict[str, float]):
    """Create parameter importance chart."""
    if not importance_data:
        return None
    
    df = pd.DataFrame(list(importance_data.items()), columns=['Parameter', 'Importance'])
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Parameter', 
        orientation='h',
        title="Parameter Importance",
        labels={'Importance': 'Importance Score', 'Parameter': 'Parameter'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title="Parameter"
    )
    
    return fig

def create_convergence_chart(convergence_history: list):
    """Create convergence history chart."""
    if not convergence_history:
        return None
    
    df = pd.DataFrame({
        'Trial': range(1, len(convergence_history) + 1),
        'Separation Angle': convergence_history
    })
    
    fig = px.line(
        df,
        x='Trial',
        y='Separation Angle',
        title="Optimization Convergence",
        labels={'Separation Angle': 'Separation Angle (degrees)', 'Trial': 'Trial Number'}
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Trial Number",
        yaxis_title="Separation Angle (degrees)"
    )
    
    return fig

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 DLD Geometry Optimization Tool</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Backend API is not available. Please ensure the backend service is running.")
        st.info("To start the backend, run: `python -m backend.api.app`")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Input Parameters")
    
    # Get default parameters
    defaults = get_default_parameters()
    
    # Cell Parameters
    st.sidebar.subheader("📊 Cell Parameters")
    DI1 = st.sidebar.number_input("Deformation Index of Cell 1 (DI1)", 
                                 min_value=0.0, max_value=1.0, value=defaults["DI1"])
    DI2 = st.sidebar.number_input("Deformation Index of Cell 2 (DI2)", 
                                 min_value=0.0, max_value=1.0, value=defaults["DI2"])
    R1 = st.sidebar.number_input("Radius of Cell 1 (R1)", 
                                min_value=0.0, max_value=1.0, value=defaults["R1"])
    R2 = st.sidebar.number_input("Radius of Cell 2 (R2)", 
                                min_value=0.0, max_value=1.0, value=defaults["R2"])
    
    # DLD Parameters
    st.sidebar.subheader("🔧 DLD Parameters")
    
    P_min, P_max = st.sidebar.slider(
        "Pillar (P) Radius (μm)", 
        min_value=0.0, 
        max_value=50.0, 
        value=(round(defaults["P_min"], 1), round(defaults["P_max"], 1)),
        step=0.1,
        format="%.1f"
    )
    
    Gh_min, Gh_max = st.sidebar.slider(
        "Horizontal Gap (Gh) (μm)", 
        min_value=0.0, 
        max_value=50.0, 
        value=(round(defaults["Gh_min"], 1), round(defaults["Gh_max"], 1)),
        step=0.1,
        format="%.1f"
    )
    
    Gv_min, Gv_max = st.sidebar.slider(
        "Vertical Gap (Gv) (μm)", 
        min_value=0.0, 
        max_value=50.0, 
        value=(round(defaults["Gv_min"], 1), round(defaults["Gv_max"], 1)),
        step=0.1,
        format="%.1f"
    )
    
    alpha_min, alpha_max = st.sidebar.slider(
        "Row Shift Angle (α) (deg)", 
        min_value=0.0, 
        max_value=20.0, 
        value=(round(defaults["alpha_min"], 1), round(defaults["alpha_max"], 1)),
        step=0.1,
        format="%.1f"
    )
    
    Q_min, Q_max = st.sidebar.slider(
        "Flow Rate (Q) (μL/min)", 
        min_value=0.0, 
        max_value=20.0, 
        value=(round(defaults["Q_min"], 1), round(defaults["Q_max"], 1)),
        step=0.1,
        format="%.1f"
    )
    
    # Optimization Settings
    st.sidebar.subheader("⚙️ Optimization Settings")
    n_trials = st.sidebar.slider("Number of Trials", 
                                min_value=10, max_value=200, value=defaults["n_trials"])
    n_startup_trials = st.sidebar.slider("Startup Trials (Random)", 
                                        min_value=5, max_value=30, value=defaults["n_startup_trials"])
    
    # Optimize button
    optimize_button = st.sidebar.button("🚀 Optimize DLD Geometry", type="primary")
    
    # Main content area
    if optimize_button:
        st.markdown("## 🔄 Running Optimization...")
        
        # Prepare parameters
        parameters = {
            "DI1": DI1, "DI2": DI2, "R1": R1, "R2": R2,
            "P_min": P_min, "P_max": P_max,
            "Gh_min": Gh_min, "Gh_max": Gh_max,
            "Gv_min": Gv_min, "Gv_max": Gv_max,
            "alpha_min": alpha_min, "alpha_max": alpha_max,
            "Q_min": Q_min, "Q_max": Q_max,
            "n_trials": n_trials, "n_startup_trials": n_startup_trials,
            "random_state": defaults["random_state"]
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run optimization
        with st.spinner("Optimizing DLD geometry parameters..."):
            result = run_optimization(parameters)
        
        if result:
            progress_bar.progress(100)
            status_text.text("✅ Optimization completed!")
            
            st.markdown("## 📊 Optimization Results")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Optimal Parameters")
                st.metric("Post Size (P)", f"{result['optimal_P']:.2f} μm")
                st.metric("Horizontal Gap (Gh)", f"{result['optimal_Gh']:.2f} μm")
                st.metric("Vertical Gap (Gv)", f"{result['optimal_Gv']:.2f} μm")
                st.metric("Row Shift (α)", f"{result['optimal_alpha']:.2f}°")
                st.metric("Flow Rate (Q)", f"{result['optimal_Q']:.2f} μL/min")
            
            with col2:
                st.markdown("### 📈 Performance Metrics")
                st.metric("Max Separation Angle", f"{result['max_separation_angle']:.4f}°")
                st.metric("Optimization Time", f"{result['optimization_time']:.2f}s")
                st.metric("Trials Completed", result['n_trials'])
                st.metric("Best Trial", result['best_trial_number'])
                st.metric("Trials/min", f"{result['n_trials']/(result['optimization_time']/60):.1f}")
            
            # Charts
            st.markdown("## 📊 Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if result.get('parameter_importance'):
                    fig_importance = create_parameter_importance_chart(result['parameter_importance'])
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                if result.get('convergence_history'):
                    fig_convergence = create_convergence_chart(result['convergence_history'])
                    if fig_convergence:
                        st.plotly_chart(fig_convergence, use_container_width=True)
            
            # Download results
            st.markdown("## 💾 Download Results")
            
            # Create results summary
            results_summary = {
                "optimization_parameters": parameters,
                "optimal_parameters": {
                    "P": result['optimal_P'],
                    "Gh": result['optimal_Gh'],
                    "Gv": result['optimal_Gv'],
                    "alpha": result['optimal_alpha'],
                    "Q": result['optimal_Q']
                },
                "results": {
                    "max_separation_angle": result['max_separation_angle'],
                    "optimization_time": result['optimization_time'],
                    "n_trials": result['n_trials'],
                    "best_trial_number": result['best_trial_number']
                },
                "parameter_importance": result.get('parameter_importance', {}),
                "timestamp": result.get('timestamp', '')
            }
            
            # JSON download
            st.download_button(
                label="📄 Download Results (JSON)",
                data=json.dumps(results_summary, indent=2),
                file_name="dld_optimization_results.json",
                mime="application/json"
            )
            
            # CSV download for parameter importance
            if result.get('parameter_importance'):
                importance_df = pd.DataFrame(
                    list(result['parameter_importance'].items()),
                    columns=['Parameter', 'Importance']
                )
                st.download_button(
                    label="📊 Download Parameter Importance (CSV)",
                    data=importance_df.to_csv(index=False),
                    file_name="parameter_importance.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome message
        st.markdown("""
        ## 🎯 Welcome to DLD Optimization Tool
        
        This tool optimizes DLD (Deterministic Lateral Displacement) geometry parameters to maximize 
        the separation angle difference between two cells with different deformation indices.
        
        ### 🚀 How to use:
        1. **Set Parameters**: Adjust the cell parameters and DLD parameters in the sidebar
        2. **Configure Optimization**: Choose the number of trials and startup trials
        3. **Run Optimization**: Click the "Optimize DLD Geometry" button
        4. **Analyze Results**: View optimal parameters, performance metrics, and visualizations
        
        ### 🔬 About the Optimization:
        - Uses **Optuna's TPE (Tree-structured Parzen Estimator)** sampler
        - Considers parameter correlations for efficient exploration
        - Provides parameter importance analysis
        - Shows convergence history for optimization monitoring
        
        ### 📊 Key Features:
        - **Real-time Progress**: Track optimization progress
        - **Parameter Validation**: Automatic validation of input parameters
        - **Visualization**: Interactive charts for results analysis
        - **Export Results**: Download results in JSON and CSV formats
        """)
        
        # Show API status
        if api_healthy:
            st.success("✅ Backend API is running and ready for optimization!")

if __name__ == "__main__":
    main() 
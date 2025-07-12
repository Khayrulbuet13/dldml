import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import optuna
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.base import BaseConfig

# Configuration
config = BaseConfig()
API_BASE_URL = f"http://localhost:{config.API_PORT}/api/v1"

# Page configuration
st.set_page_config(
    page_title="DLD Optimization Tool",
    page_icon="🔬",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "study" not in st.session_state:
    st.session_state.study = None
if "best_params" not in st.session_state:
    st.session_state.best_params = None
if "best_separation" not in st.session_state:
    st.session_state.best_separation = None
if "optimization_time" not in st.session_state:
    st.session_state.optimization_time = None
if "study_data" not in st.session_state:
    st.session_state.study_data = None
if "trials_dataframe" not in st.session_state:
    st.session_state.trials_dataframe = None

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

def create_optuna_study_from_data(study_data: Dict[str, Any], trials_dataframe: Dict[str, Any]) -> Optional[Any]:
    """Create an Optuna study object from the data received from the API."""
    try:
        # Create a real Optuna study
        study = optuna.create_study(direction='minimize')
        
        # Populate the study with trials from the dataframe
        if trials_dataframe and trials_dataframe['data']:
            df = pd.DataFrame(trials_dataframe['data'], columns=trials_dataframe['columns'])
            
            # First pass: collect all parameter ranges to create consistent distributions
            param_ranges = {}
            for col in df.columns:
                if col.startswith('params_'):
                    param_name = col.replace('params_', '')
                    values = df[col].dropna()
                    if len(values) > 0:
                        param_ranges[param_name] = {
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
            
            # Second pass: create trials with consistent distributions
            for i, row in df.iterrows():
                # Extract parameters
                params = {}
                for col in df.columns:
                    if col.startswith('params_'):
                        param_name = col.replace('params_', '')
                        params[param_name] = row[col]
                
                # Extract value
                value = row['value'] if 'value' in df.columns else 0.0
                
                # Create distributions for parameters using the collected ranges
                distributions = {}
                for param_name, param_value in params.items():
                    if param_name in param_ranges:
                        # Use the actual parameter range from the data
                        param_range = param_ranges[param_name]
                        distributions[param_name] = optuna.distributions.FloatDistribution(
                            low=param_range['min'],
                            high=param_range['max']
                        )
                    else:
                        # Fallback: create a simple uniform distribution around the value
                        distributions[param_name] = optuna.distributions.FloatDistribution(
                            low=param_value * 0.9, 
                            high=param_value * 1.1
                        )
                
                # Create a trial
                trial = optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=value
                )
                
                # Add trial to study
                study.add_trial(trial)
        
        return study
    except Exception as e:
        st.error(f"Error creating study object: {e}")
        return None

# Function to display results
def show_results(study, best_params, best_separation, optimization_time, study_data, trials_dataframe):
    st.markdown("## 📊 Optimization Results")
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Optimal Parameters")
        st.metric("Post Size (P)", f"{best_params['P']:.2f} μm")
        st.metric("Horizontal Gap (Gh)", f"{best_params['Gh']:.2f} μm")
        st.metric("Vertical Gap (Gv)", f"{best_params['Gv']:.2f} μm")
        st.metric("Row Shift (α)", f"{best_params['alpha']:.2f}°")
        st.metric("Flow Rate (Q)", f"{best_params['Q']:.2f} μL/min")
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.metric("Max Separation Angle", f"{best_separation:.4f}°")
        st.metric("Optimization Time", f"{optimization_time:.2f}s")
        if study and hasattr(study, 'trials'):
            st.metric("Trials Completed", len(study.trials))
            st.metric("Best Trial", study.best_trial.number if study.best_trial else 0)
            st.metric("Trials/min", f"{len(study.trials)/(optimization_time/60):.1f}")
        else:
            st.metric("Trials Completed", "N/A")
            st.metric("Best Trial", "N/A")
            st.metric("Trials/min", "N/A")

    # Parameter importance analysis
    st.markdown("## 📊 Analysis")
    
    st.subheader("Parameter Importance:")
    try:
        if study and hasattr(study, 'trials') and len(study.trials) > 0:
            param_importance = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame(list(param_importance.items()), columns=['Parameter', 'Importance'])
            st.bar_chart(importance_df.set_index('Parameter'))
        else:
            st.write("Parameter importance not available - insufficient trial data")
    except Exception as e:
        st.write(f"Could not compute parameter importance: {e}")

    # 2. Parameter Interaction Matrix
    st.subheader("Parameter Interactions")
    try:
        if study and hasattr(study, 'trials') and len(study.trials) > 0:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.update_layout(
                title='Parameter Interactions and Objective Values',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Parameter interactions plot not available - insufficient trial data")
    except Exception as e:
        st.error(f"Error generating parameter interactions plot: {str(e)}")
        st.info("This might happen if there are not enough trials or if the study data is incomplete.")

    # 3. Interactive Response Surface Visualization
    st.subheader("Response Surface Analysis")
    st.markdown("""
    **Response Surface Analysis**: Shows the relationship between two selected parameters and separation performance.
    - **Contour lines**: Represent performance levels
    - **Peaks**: Optimal performance regions
    - **Valleys**: Poor performance regions
    - **Steep gradients**: High sensitivity to parameter changes
    """)
    
    try:
        if study and hasattr(study, 'trials') and len(study.trials) > 0:
            # Initialize session state for parameter selection if not exists
            if 'x_param' not in st.session_state or 'y_param' not in st.session_state:
                try:
                    param_importance = optuna.importance.get_param_importances(study)
                    param_options = list(param_importance.keys())
                    
                    # Ensure we have at least 2 parameters
                    if len(param_options) >= 2:
                        top_params = sorted(param_importance, key=param_importance.get, reverse=True)[:2]
                        st.session_state.x_param = top_params[0]
                        st.session_state.y_param = top_params[1]
                    elif len(param_options) == 1:
                        st.session_state.x_param = param_options[0]
                        st.session_state.y_param = param_options[0]  # Same parameter as fallback
                    else:
                        # Fallback if no parameters available
                        param_options = ['P', 'Gh', 'Gv', 'alpha', 'Q']
                        st.session_state.x_param = param_options[0]
                        st.session_state.y_param = param_options[1]
                except:
                    # Fallback if parameter importance fails
                    param_options = ['P', 'Gh', 'Gv', 'alpha', 'Q']
                    st.session_state.x_param = param_options[0]
                    st.session_state.y_param = param_options[1]
            
            # Get parameter importance and options
            try:
                param_importance = optuna.importance.get_param_importances(study)
                param_options = list(param_importance.keys())
            except:
                # Fallback if parameter importance fails
                param_options = ['P', 'Gh', 'Gv', 'alpha', 'Q']
            
            # Ensure we have enough parameters
            if len(param_options) < 2:
                # Use fallback parameters if we don't have enough
                param_options = ['P', 'Gh', 'Gv', 'alpha', 'Q']
            
            # Ensure current session state values are valid
            if st.session_state.x_param not in param_options:
                st.session_state.x_param = param_options[0]
            if st.session_state.y_param not in param_options:
                st.session_state.y_param = param_options[1] if len(param_options) > 1 else param_options[0]
            
            # Create selection widgets with validation
            col1, col2 = st.columns(2)
            with col1:
                x_param = st.selectbox(
                    "X-axis Parameter", 
                    param_options, 
                    index=param_options.index(st.session_state.x_param),
                    key="x_param_selector"
                )
                st.session_state.x_param = x_param
                
            with col2:
                # Filter out the X parameter from Y options to prevent same selection
                y_options = [p for p in param_options if p != x_param]
                if len(y_options) > 0:
                    current_y = st.session_state.y_param if st.session_state.y_param != x_param else y_options[0]
                    y_param = st.selectbox(
                        "Y-axis Parameter", 
                        y_options,
                        index=y_options.index(current_y) if current_y in y_options else 0,
                        key="y_param_selector"
                    )
                    st.session_state.y_param = y_param
                else:
                    st.write("No other parameters available for Y-axis")
                    return
            
            # Generate the contour plot
            if x_param != y_param:
                try:
                    fig = optuna.visualization.plot_contour(study, params=[x_param, y_param])
                    fig.update_layout(
                        title=f'Response Surface: {x_param} vs {y_param}',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as plot_error:
                    st.error(f"Error generating contour plot: {str(plot_error)}")
                    st.info("Try selecting different parameters or running more optimization trials.")
            else:
                st.warning("Please select different parameters for X and Y axes")
                # Show a placeholder or previous plot to prevent hanging
                st.info("Select different parameters above to view the response surface.")
        else:
            st.write("Response surface analysis not available - insufficient trial data")
            
    except Exception as e:
        st.error(f"Could not generate surface plot: {str(e)}")
        st.info("This might happen if there are not enough trials or if parameter importance cannot be calculated.")

    # Optimization convergence summary
    st.subheader("Optimization Summary:")
    if study and hasattr(study, 'best_value'):
        st.write(f"**Best objective value found:** {study.best_value:.4f}")
    if study and hasattr(study, 'trials'):
        st.write(f"**Number of trials:** {len(study.trials)}")
    st.write("**Optimization completed successfully!**")
    
    # Download results
    st.markdown("## 💾 Download Results")
    
    # Create results summary
    results_summary = {
        "optimization_parameters": {
            "DI1": st.session_state.get('DI1', 0.5), 
            "DI2": st.session_state.get('DI2', 0.8), 
            "R1": st.session_state.get('R1', 0.5), 
            "R2": st.session_state.get('R2', 0.8),
            "P_min": st.session_state.get('P_min', 5.0), 
            "P_max": st.session_state.get('P_max', 15.0),
            "Gh_min": st.session_state.get('Gh_min', 5.0), 
            "Gh_max": st.session_state.get('Gh_max', 15.0),
            "Gv_min": st.session_state.get('Gv_min', 5.0), 
            "Gv_max": st.session_state.get('Gv_max', 15.0),
            "alpha_min": st.session_state.get('alpha_min', 1.0), 
            "alpha_max": st.session_state.get('alpha_max', 5.0),
            "Q_min": st.session_state.get('Q_min', 0.5), 
            "Q_max": st.session_state.get('Q_max', 5.0),
            "n_trials": st.session_state.get('n_trials', 100), 
            "n_startup_trials": st.session_state.get('n_startup_trials', 15)
        },
        "optimal_parameters": {
            "P": best_params['P'],
            "Gh": best_params['Gh'],
            "Gv": best_params['Gv'],
            "alpha": best_params['alpha'],
            "Q": best_params['Q']
        },
        "results": {
            "max_separation_angle": best_separation,
            "optimization_time": optimization_time,
            "n_trials": len(study.trials) if study and hasattr(study, 'trials') else 0,
            "best_trial_number": study.best_trial.number if study and study.best_trial else 0
        },
        "parameter_importance": optuna.importance.get_param_importances(study) if study and study.trials else {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # JSON download
    st.download_button(
        label="📄 Download Results (JSON)",
        data=json.dumps(results_summary, indent=2),
        file_name="dld_optimization_results.json",
        mime="application/json"
    )
    
    # CSV download for optimal parameters
    optimal_params_df = pd.DataFrame([
        {"Parameter": "Post Size (P)", "Value": best_params['P'], "Unit": "μm"},
        {"Parameter": "Horizontal Gap (Gh)", "Value": best_params['Gh'], "Unit": "μm"},
        {"Parameter": "Vertical Gap (Gv)", "Value": best_params['Gv'], "Unit": "μm"},
        {"Parameter": "Row Shift Angle (α)", "Value": best_params['alpha'], "Unit": "°"},
        {"Parameter": "Flow Rate (Q)", "Value": best_params['Q'], "Unit": "μL/min"}
    ])
    
    st.download_button(
        label="📊 Download Optimal Parameters (CSV)",
        data=optimal_params_df.to_csv(index=False),
        file_name="optimal_parameters.csv",
        mime="text/csv"
    )

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
    st.sidebar.header("Input Parameters")
    
    # Get default parameters
    defaults = get_default_parameters()
    
    # Cell Parameters
    st.sidebar.subheader("📊 Cell Parameters")
    DI1 = st.sidebar.number_input("Deformation Index of Cell 1 (DI1)", min_value=0.0, max_value=1.0, value=defaults["DI1"])
    DI2 = st.sidebar.number_input("Deformation Index of Cell 2 (DI2)", min_value=0.0, max_value=1.0, value=defaults["DI2"])
    R1 = st.sidebar.number_input("Radius of Cell 1 (R1)", min_value=0.0, max_value=1.0, value=defaults["R1"])
    R2 = st.sidebar.number_input("Radius of Cell 2 (R2)", min_value=0.0, max_value=1.0, value=defaults["R2"])
    
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
    
    # Optimization parameters
    st.sidebar.subheader("⚙️ Optimization Settings")
    n_trials = st.sidebar.slider("Number of Trials", min_value=10, max_value=200, value=defaults["n_trials"])
    n_startup_trials = st.sidebar.slider("Startup Trials (Random)", min_value=5, max_value=30, value=defaults["n_startup_trials"])
    
    optimize_button = st.sidebar.button("🚀 Optimize DLD Geometry", type="primary")
    
    # Run optimization when button is pressed
    if optimize_button:
        # Clear previous parameter selections to reset the interface
        if 'x_param' in st.session_state:
            del st.session_state.x_param
        if 'y_param' in st.session_state:
            del st.session_state.y_param
        
        st.markdown("## 🔄 Running Optimization...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
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
        
        # Store parameters in session state for download
        for key, value in parameters.items():
            st.session_state[key] = value
        
        # Run optimization with progress tracking
        with st.spinner("Optimizing DLD geometry parameters..."):
            result = run_optimization(parameters)
        
        if result:
            progress_bar.progress(100)
            status_text.text("✅ Optimization completed!")
            
            # Extract results
            best_params = {
                'P': result['optimal_P'],
                'Gh': result['optimal_Gh'],
                'Gv': result['optimal_Gv'],
                'alpha': result['optimal_alpha'],
                'Q': result['optimal_Q']
            }
            best_separation = result['max_separation_angle']
            optimization_time = result['optimization_time']
            
            # Create study object for visualizations
            study = create_optuna_study_from_data(result['study_data'], result['trials_dataframe'])
            
            # Store results in session state
            st.session_state.study = study
            st.session_state.best_params = best_params
            st.session_state.best_separation = best_separation
            st.session_state.optimization_time = optimization_time
            st.session_state.study_data = result['study_data']
            st.session_state.trials_dataframe = result['trials_dataframe']
            
            # Show results
            show_results(study, best_params, best_separation, optimization_time, result['study_data'], result['trials_dataframe'])
    
    # Show stored results on rerun
    elif st.session_state.study is not None:
        show_results(
            st.session_state.study, 
            st.session_state.best_params, 
            st.session_state.best_separation, 
            st.session_state.optimization_time,
            st.session_state.study_data,
            st.session_state.trials_dataframe
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
        - Provides comprehensive parameter importance analysis
        - Shows detailed convergence history for optimization monitoring
        
        ### 📊 Advanced Visualizations:
        - **📈 Parameter Importance**: Bar chart showing which parameters most affect separation performance
        - **📈 Optimization Progress**: Real-time progress tracking with confidence bands
        - **🔄 Parameter Interactions**: Parallel coordinates plot revealing parameter relationships
        - **🎯 Response Surface Analysis**: Interactive contour plot showing how two parameters affect performance
        
        ### 📊 Key Features:
        - **Real-time Progress**: Track optimization progress
        - **Parameter Validation**: Automatic validation of input parameters
        - **Advanced Visualizations**: Interactive charts for comprehensive results analysis
        - **Session State Management**: Smooth parameter switching without hanging
        - **Export Results**: Download results in JSON and CSV formats
        """)
        
        # Show API status
        if api_healthy:
            st.success("✅ Backend API is running and ready for optimization!")

if __name__ == "__main__":
    main() 
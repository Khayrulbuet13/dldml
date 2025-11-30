#!/usr/bin/env python3
"""
Startup script for the backend that ensures the OptimizationModel class is available.
"""

import sys
import os

# Define the OptimizationModel class in __main__ module
class OptimizationModel:
    def __init__(self, xgb_model, scaler, selector, feature_names):
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.selector = selector
        self.feature_names = feature_names
    
    def predict(self, X):
        """
        Predict migration angle theta_m for input features.
        X should be a 2D array with columns: [DI, Pr, Pg, alpha]
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection
        X_selected = self.selector.transform(X_scaled)
        
        # Convert to DMatrix
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X_selected)
        
        # Predict
        predictions = self.xgb_model.predict(dmatrix)
        
        return predictions

# Make sure the class is available in __main__
if '__main__' not in sys.modules:
    sys.modules['__main__'] = sys.modules[__name__]

# Now start the backend
if __name__ == "__main__":
    import uvicorn
    from backend.api.app import app
    
    print("🚀 Starting DLD Optimization Backend...")
    print("✅ OptimizationModel class defined in __main__")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 
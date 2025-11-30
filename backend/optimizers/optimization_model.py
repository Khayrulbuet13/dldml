import numpy as np
import xgboost as xgb

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
        
        Where:
            DI: Deformation Index [0-1]
            Pr: Pillar radius (μm)
            Pg: Pillar gap (μm)
            alpha: Row shift angle (degrees, continuous representation of N_p)
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection (if selector exists)
        if self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X_selected)
        
        # Predict
        predictions = self.xgb_model.predict(dmatrix)
        
        return predictions 
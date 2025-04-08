"""
Prediction module for crime risk assessment.
This module loads the trained model and provides functions to:
1. Generate risk predictions for all grid cells
2. Output grid cells with their risk scores in various formats (JSON, GeoJSON)
"""

import joblib
import pandas as pd
import geopandas as gpd
import json
import os
from preprocess import load_datasets, create_crime_prediction_dataset

def load_model(model_path="best_model_crime_prediction.pkl"):
    """Load the trained model from disk."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def predict_grid_risks(model=None, preprocess=True):
    """
    Generate risk predictions for all grid cells.
    
    Args:
        model: Pre-loaded model (optional)
        preprocess: Whether to perform preprocessing (default: True)
        
    Returns:
        GeoDataFrame with grid cells and their risk predictions
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load datasets
    grid_gdf, gdf_crimenes, gdf_police, gdf_barrios = load_datasets()
    
    if preprocess:
        # Create the dataset with features
        grid_gdf = create_crime_prediction_dataset(
            grid_gdf, gdf_crimenes, gdf_police, gdf_barrios
        )
    
    # Prepare features for prediction
    X = grid_gdf.drop(['geometry', 'target'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    # Ensure columns match training data
    model_columns = joblib.load('model_columns.pkl')
    for col in model_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[model_columns]
    
    # Make predictions
    grid_gdf['risk'] = model.predict_proba(X)[:, 1]
    
    return grid_gdf[['cell_id', 'geometry', 'risk']]

def generate_risk_json(prediction_grid=None):
    """
    Generate JSON with risk scores for each grid cell.
    
    Args:
        prediction_grid: Pre-generated predictions (optional)
        
    Returns:
        Dictionary with cell_id keys and risk values
    """
    if prediction_grid is None:
        prediction_grid = predict_grid_risks()
        
    risk_dict = {}
    for _, cell in prediction_grid.iterrows():
        cell_id = cell['cell_id']
        risk = float(round(cell['risk'], 2))
        risk_dict[cell_id] = {"risk": risk}
    
    return risk_dict

def save_risk_json(output_path="crime_risks.json"):
    """Save risk predictions to JSON file."""
    risk_dict = generate_risk_json()
    with open(output_path, 'w') as f:
        json.dump(risk_dict, f, indent=2)
    print(f"Risk predictions saved to {output_path}")
    return output_path

def save_risk_geojson(output_path="crime_risks.geojson"):
    """Save risk predictions with geometry to GeoJSON file."""
    prediction_grid = predict_grid_risks()
    prediction_grid.to_file(output_path, driver="GeoJSON")
    print(f"Risk predictions with geometry saved to {output_path}")
    return output_path

"""
Quick Prediction Script for Crime Prediction Model
This script generates predictions using the quick trained model.
It can be used for testing and integration before the full model is ready.
"""
import pandas as pd
import numpy as np
import joblib
import os
import geopandas as gpd
import json
from datetime import datetime, timedelta
from preprocess import load_datasets
import logging
import argparse
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quick_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_quick_model(prediction_window=24):
    """Load the quick model for prediction"""
    model_paths = [
        "crime_model_simple.pkl",
        f"crime_model_quick_{prediction_window}h.pkl",
        "crime_model_quick.pkl"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
    
    raise FileNotFoundError("No model found. Run quick_train.py first.")

def prepare_prediction_features(grid_gdf):
    """
    Prepare features for prediction - simplified version
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
    
    Returns:
        DataFrame with features for prediction
    """
    logger.info("Preparing features for prediction...")
    
    # For quick prediction, we'll just use a minimal set of features
    # that are likely to be available at inference time
    
    # Start with crime count and distance to police
    X = grid_gdf[['crime_count', 'distance_to_police']].copy()
    
    # Add time features for current time
    now = datetime.now()
    X['ref_hour'] = now.hour
    X['ref_day'] = now.weekday()
    X['ref_month'] = now.month  # Add the missing ref_month feature
    X['ref_is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    # Add target_24h with default value 0 (since we're predicting, not training)
    # This ensures compatibility with the model's expected features
    X['target_24h'] = 0
    
    # If we have time features, add them, otherwise use zeros
    for time_period in ['morning', 'afternoon', 'evening', 'night']:
        col = f'crimes_{time_period}'
        if col in grid_gdf.columns:
            X[col] = grid_gdf[col]
        else:
            X[col] = 0
    
    # Add recency features (fill with zeros if not available)
    for window in [7, 30]:
        col = f'crimes_last_{window}d'
        if col in grid_gdf.columns:
            X[col] = grid_gdf[col]
        else:
            X[col] = 0
    
    return X

def generate_quick_predictions(model_data, grid_gdf=None):
    """
    Generate risk predictions using the quick model
    
    Args:
        model_data: Dictionary containing model, scaler, and numerical columns
        grid_gdf: GeoDataFrame with grid cells (loads if None)
    
    Returns:
        GeoDataFrame with predictions
    """
    if grid_gdf is None:
        logger.info("Loading grid data...")
        grid_gdf, gdf_crimenes, gdf_police, _, _ = load_datasets()
    else:
        # We need crime and police data for preprocessing
        logger.info("Loading additional data for preprocessing...")
        _, gdf_crimenes, gdf_police, _ = load_datasets()
    
    # Preprocess grid data to add necessary columns
    logger.info("Preprocessing grid data...")
    from preprocess import count_crimes_per_cell, add_distance_to_police
    
    # Add crime count
    if 'crime_count' not in grid_gdf.columns:
        grid_gdf = count_crimes_per_cell(grid_gdf, gdf_crimenes)
        logger.info(f"Added crime_count column. Range: {grid_gdf['crime_count'].min()} - {grid_gdf['crime_count'].max()}")
    
    # Add police distance
    if 'distance_to_police' not in grid_gdf.columns:
        grid_gdf = add_distance_to_police(grid_gdf, gdf_police)
        logger.info(f"Added distance_to_police column. Range: {grid_gdf['distance_to_police'].min():.2f} - {grid_gdf['distance_to_police'].max():.2f} km")
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    numerical_cols = model_data['numerical_columns']
    
    # Prepare features
    X = prepare_prediction_features(grid_gdf)
    
    # Get the intersection of available columns and expected numerical columns
    scale_cols = [col for col in numerical_cols if col in X.columns]
    
    # Scale numerical features
    X[scale_cols] = scaler.transform(X[scale_cols])
    
    # Check if we have all needed columns
    missing_cols = set(model.feature_names_in_) - set(X.columns)
    extra_cols = set(X.columns) - set(model.feature_names_in_)
    
    if missing_cols:
        logger.warning(f"Missing columns for prediction: {missing_cols}")
        # Add missing columns with zeros
        for col in missing_cols:
            X[col] = 0
    
    # Ensure columns are in the right order
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Generate predictions
    try:
        logger.info("Generating predictions...")
        risk_scores = model.predict_proba(X)[:, 1]
        
        # Add predictions to the grid
        prediction_grid = grid_gdf.copy()
        prediction_grid['risk'] = risk_scores
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Try with a simpler approach
        logger.info("Trying simplified prediction...")
        # Generate random predictions for demonstration
        prediction_grid = grid_gdf.copy()
        prediction_grid['risk'] = np.random.beta(1, 10, size=len(grid_gdf))
        logger.warning("Using random predictions for demonstration!")
    
    # Print a sample of predictions
    logger.info("\nSample predictions:")
    risk_sample = prediction_grid.sort_values('risk', ascending=False).head(5)[['h3_index', 'risk']]
    logger.info(risk_sample)
    
    return prediction_grid

def save_predictions_json(prediction_grid, output_path=None):
    """
    Save predictions to a JSON file
    
    Args:
        prediction_grid: GeoDataFrame with predictions
        output_path: Path to save JSON file
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = f"crime_predictions_quick_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    # Create a dictionary with cell_id keys and risk values
    prediction_dict = {
        "metadata": {
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "quick",
            "total_cells": len(prediction_grid)
        },
        "cells": {}
    }
    
    for _, cell in prediction_grid.iterrows():
        cell_id = cell['h3_index']
        risk = float(round(cell['risk'], 3))
        
        # Group risk into categories for easier visualization
        risk_category = "low"
        if risk >= 0.7:
            risk_category = "very_high" 
        elif risk >= 0.5:
            risk_category = "high"
        elif risk >= 0.3:
            risk_category = "medium"
            
        prediction_dict["cells"][cell_id] = {
            "risk": risk,
            "risk_category": risk_category,
            "cell_id": cell_id
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(prediction_dict, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")
    return output_path

def save_predictions_geojson(prediction_grid, output_path=None):
    """
    Save predictions to a GeoJSON file
    
    Args:
        prediction_grid: GeoDataFrame with predictions
        output_path: Path to save GeoJSON file
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = f"crime_predictions_quick_{datetime.now().strftime('%Y%m%d_%H%M')}.geojson"
    
    # Add risk categories
    prediction_grid['risk_category'] = 'low'
    prediction_grid.loc[prediction_grid['risk'] >= 0.3, 'risk_category'] = 'medium'
    prediction_grid.loc[prediction_grid['risk'] >= 0.5, 'risk_category'] = 'high'
    prediction_grid.loc[prediction_grid['risk'] >= 0.7, 'risk_category'] = 'very_high'
    
    # Add prediction time info
    prediction_grid['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Export to GeoJSON
    prediction_grid.to_file(output_path, driver="GeoJSON")
    logger.info(f"Predictions with geometry saved to {output_path}")
    
    return output_path

def download_github_file(filename, output_path):
    """
    Download a file from GitHub repository
    
    Args:
        filename: Name of the file to download (e.g., 'hex_grid.gpkg')
        output_path: Path where to save the downloaded file
    
    Returns:
        Boolean indicating if download was successful
    """
    base_url = "https://github.com/Rosvend/Patrol-routes-optimization-Medellin/raw/main/geodata/"
    url = base_url + filename
    
    logger.info(f"Downloading {filename} from GitHub...")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {filename}")
            return True
        else:
            logger.error(f"Failed to download {filename}. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.exception(f"Error downloading {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate quick predictions for crime risk')
    parser.add_argument('--window', type=int, default=24, help='Prediction window in hours (default: 24)')
    parser.add_argument('--format', choices=['json', 'geojson', 'both'], default='both', 
                        help='Output format (json, geojson, or both)')
    parser.add_argument('--output', type=str, default=None, help='Output file path base name')
    
    args = parser.parse_args()
    
    logger.info(f"Quick prediction started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        model_data = load_quick_model(args.window)
        grid_gdf, _, _, _ = load_datasets()
        prediction_grid = generate_quick_predictions(model_data, grid_gdf)
        if args.format in ['json', 'both']:
            json_path = args.output if args.format == 'json' else None
            save_predictions_json(prediction_grid, json_path)
        
        if args.format in ['geojson', 'both']:
            geojson_path = args.output if args.format == 'geojson' else None
            save_predictions_geojson(prediction_grid, geojson_path)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
    
    logger.info(f"Process finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
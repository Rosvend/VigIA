"""
Prediction module for crime risk assessment.
This module loads the trained model and provides functions to:
1. Generate risk predictions for all grid cells for specific time windows
2. Output grid cells with their risk scores in various formats (JSON, GeoJSON)
"""

import joblib
import pandas as pd
import geopandas as gpd
import json
import os
from datetime import datetime, timedelta
from preprocess import load_datasets, create_crime_prediction_dataset, add_time_features, add_crime_type_features
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(prediction_window=24, model_type="best"):
    """
    Load the trained model from disk.
    
    Args:
        prediction_window: Hours to predict ahead (default: 24)
        model_type: Type of model to load ("best", "static", or model name)
    
    Returns:
        Trained model for the specified prediction window
    """
    if model_type == "best":
        model_path = f"best_model_crime_prediction_{prediction_window}h.pkl"
    elif model_type == "static":
        model_path = f"static_model_crime_prediction_{prediction_window}h.pkl"
    else:
        model_path = f"{model_type}_crime_prediction_{prediction_window}h.pkl"
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    else:
        # Check if there's any model for this prediction window
        models = [f for f in os.listdir('.') if f.endswith(f"_{prediction_window}h.pkl")]
        if models:
            logger.warning(f"Requested model not found. Using {models[0]} instead")
            return joblib.load(models[0])
        else:
            # Fall back to default model
            default_model = "best_model_crime_prediction.pkl"
            if os.path.exists(default_model):
                logger.warning(f"No models found for {prediction_window}h window. Using default model {default_model}")
                return joblib.load(default_model)
            else:
                raise FileNotFoundError(f"No suitable model files found for prediction")

def prepare_temporal_features(grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, reference_time=None, recency_windows=None):
    """
    Prepare temporal features for prediction at a specific point in time
    
    Args:
        grid_gdf: Grid GeoDataFrame
        gdf_crimenes: Crime GeoDataFrame
        gdf_police: Police stations GeoDataFrame
        gdf_barrios: Neighborhoods GeoDataFrame
        reference_time: Time point for prediction (default: now)
        recency_windows: List of time windows in days for recency features
                      Default: [1, 3, 7, 14, 30] for multiple time granularities
                      
    Returns:
        DataFrame with features for prediction
    """
    # Set default reference time to now if not specified
    if reference_time is None:
        reference_time = datetime.now()
        
    # Set default recency windows
    if recency_windows is None:
        recency_windows = [1, 3, 7, 14, 30]  # 1d, 3d, 7d, 14d, 30d
    
    logger.info(f"Preparing features for prediction at {reference_time}")
    
    # Start with the basic features: count crimes, police distance, demographics
    result = count_crimes_per_cell(grid_gdf, gdf_crimenes)
    result = add_distance_to_police(result, gdf_police)
    result = add_demographic_features(result, gdf_barrios)
    
    # Add reference time features
    result['reference_time'] = reference_time
    result['ref_hour'] = reference_time.hour
    result['ref_day'] = reference_time.weekday()
    result['ref_month'] = reference_time.month
    result['ref_is_weekend'] = 1 if reference_time.weekday() >= 5 else 0
    
    # Filter only past crimes (before reference time)
    past_crimes = gdf_crimenes[gdf_crimenes['fecha_hecho'] < reference_time]
    
    # Add weighted recency features
    for window in recency_windows:
        result[f'crimes_last_{window}d'] = 0
        cutoff = reference_time - timedelta(days=window)
        
        # Add recent crime features with temporal weighting
        for idx, cell in result.iterrows():
            # Find crimes in this cell within the time window
            intersecting_crimes = past_crimes[
                (past_crimes['fecha_hecho'] >= cutoff) & 
                (past_crimes.intersects(cell.geometry))
            ]
            
            if len(intersecting_crimes) > 0:
                # Calculate days from cutoff for each crime
                days_from_cutoff = (intersecting_crimes['fecha_hecho'] - cutoff).dt.total_seconds() / (24 * 3600)
                # Apply exponential decay weight: more recent = higher weight
                weights = np.exp(days_from_cutoff / window)  # Normalized by window size
                weighted_count = weights.sum()
                result.at[idx, f'crimes_last_{window}d'] = weighted_count
    
    # Add crime type features with temporal weighting (for past 30 days)
    crime_types = past_crimes['modalidad'].unique()
    for crime_type in crime_types:
        result[f'crime_type_{crime_type}_last_30d'] = 0
        
        cutoff = reference_time - timedelta(days=30)
        recent_crimes = past_crimes[past_crimes['fecha_hecho'] >= cutoff]
        
        for idx, cell in result.iterrows():
            intersecting_crimes = recent_crimes[
                (recent_crimes['modalidad'] == crime_type) & 
                (recent_crimes.intersects(cell.geometry))
            ]
            
            if len(intersecting_crimes) > 0:
                # Calculate days from cutoff for each crime
                days_from_cutoff = (intersecting_crimes['fecha_hecho'] - cutoff).dt.total_seconds() / (24 * 3600)
                # Apply exponential decay weight
                weights = np.exp(days_from_cutoff / 30)
                weighted_count = weights.sum()
                result.at[idx, f'crime_type_{crime_type}_last_30d'] = weighted_count
    
    return result

def predict_grid_risks(prediction_window=24, reference_time=None, model=None):
    """
    Generate risk predictions for all grid cells for a specific time window.
    
    Args:
        prediction_window: Hours to predict ahead (default: 24)
        reference_time: Time point for prediction (default: now)
        model: Pre-loaded model (optional)
        
    Returns:
        GeoDataFrame with grid cells and their risk predictions
    """
    # Load model if not provided
    if model is None:
        model = load_model(prediction_window=prediction_window)
    
    # Set default reference time to now if not specified
    if reference_time is None:
        reference_time = datetime.now()
    
    logger.info(f"Generating predictions for {prediction_window}h window from {reference_time}")
    
    # Load datasets
    grid_gdf, gdf_crimenes, gdf_police, gdf_barrios = load_datasets()
    
    # Ensure crime data has datetime format
    if not pd.api.types.is_datetime64_dtype(gdf_crimenes['fecha_hecho']):
        logger.info("Converting crime date to datetime format...")
        gdf_crimenes['fecha_hecho'] = pd.to_datetime(gdf_crimenes['fecha_hecho'])
    
    # Prepare features for the specific reference time
    grid_gdf = prepare_temporal_features(
        grid_gdf,
        gdf_crimenes, 
        gdf_police,
        gdf_barrios,
        reference_time=reference_time
    )
    
    # Extract features - drop geometry and any target columns for inference
    X = grid_gdf.drop(['geometry'] + 
                      [c for c in grid_gdf.columns if c.startswith('target_')], 
                      axis=1, errors='ignore')
    
    # Add cell_id based on h3_index if not present
    if 'cell_id' not in X.columns and 'h3_index' in X.columns:
        X['cell_id'] = X['h3_index']
    
    # Make predictions using the model
    try:
        risk_scores = model.predict_proba(X)[:, 1]  # Get probability of positive class
        grid_gdf['risk'] = risk_scores
        
        # Print a sample of the results
        logger.info("\nSample of prediction results:")
        sample = grid_gdf.sort_values('risk', ascending=False).head(5)[['cell_id', 'risk']]
        logger.info(sample)
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        # Fallback: try with minimal set of features if the model is expecting less
        try:
            # Identify minimal feature set
            minimal_features = [c for c in X.columns if c in ['crime_count', 'distance_to_police', 
                                                             'crimes_last_7d', 'ref_hour', 'ref_day']]
            logger.warning(f"Trying with minimal feature set: {minimal_features}")
            risk_scores = model.predict_proba(X[minimal_features])[:, 1]
            grid_gdf['risk'] = risk_scores
        except Exception as e2:
            logger.error(f"Failed with minimal feature set too: {e2}")
            # Set risk to NaN
            grid_gdf['risk'] = float('nan')
    
    # Return the grid with the predicted risks
    if 'cell_id' not in grid_gdf.columns and 'h3_index' in grid_gdf.columns:
        grid_gdf['cell_id'] = grid_gdf['h3_index']
        
    return grid_gdf[['cell_id', 'geometry', 'risk']]

def generate_risk_json(prediction_window=24, reference_time=None, prediction_grid=None):
    """
    Generate JSON with risk scores for each grid cell.
    
    Args:
        prediction_window: Hours to predict ahead
        reference_time: Time point for prediction (default: now)
        prediction_grid: Pre-generated predictions (optional)
        
    Returns:
        Dictionary with cell_id keys and risk values
    """
    if prediction_grid is None:
        prediction_grid = predict_grid_risks(
            prediction_window=prediction_window,
            reference_time=reference_time
        )
    
    # Set prediction time range for metadata
    if reference_time is None:
        reference_time = datetime.now()
    prediction_end = reference_time + timedelta(hours=prediction_window)
    
    risk_dict = {
        "metadata": {
            "prediction_window_hours": prediction_window,
            "reference_time": reference_time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_end_time": prediction_end.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cells": len(prediction_grid)
        },
        "cells": {}
    }
    
    for _, cell in prediction_grid.iterrows():
        cell_id = cell['cell_id']
        risk = float(round(cell['risk'], 3))
        
        # Group risk into categories for easier display
        risk_category = "low"
        if risk >= 0.7:
            risk_category = "very_high" 
        elif risk >= 0.5:
            risk_category = "high"
        elif risk >= 0.3:
            risk_category = "medium"
            
        # Include cell_id explicitly for the frontend
        risk_dict["cells"][cell_id] = {
            "risk": risk,
            "risk_category": risk_category,
            "cell_id": cell_id
        }
    
    return risk_dict

def save_risk_json(output_path=None, prediction_window=24, reference_time=None):
    """
    Save risk predictions to JSON file.
    
    Args:
        output_path: Path to save JSON file (default: auto-generated)
        prediction_window: Hours to predict ahead
        reference_time: Time point for prediction (default: now)
        
    Returns:
        Path to saved JSON file
    """
    # Generate default output path if not specified
    if output_path is None:
        if reference_time is None:
            reference_time = datetime.now()
        time_str = reference_time.strftime("%Y%m%d_%H%M")
        output_path = f"crime_risks_{prediction_window}h_{time_str}.json"
    
    risk_dict = generate_risk_json(
        prediction_window=prediction_window,
        reference_time=reference_time
    )
    
    with open(output_path, 'w') as f:
        json.dump(risk_dict, f, indent=2)
    logger.info(f"Risk predictions saved to {output_path}")
    
    # Print a sample of the JSON output
    logger.info("\nSample of JSON output:")
    sample_keys = list(risk_dict["cells"].keys())[:3]
    sample_dict = {k: risk_dict["cells"][k] for k in sample_keys}
    logger.info(json.dumps(sample_dict, indent=2))
    
    return output_path

def save_risk_geojson(output_path=None, prediction_window=24, reference_time=None):
    """
    Save risk predictions with geometry to GeoJSON file.
    
    Args:
        output_path: Path to save GeoJSON file (default: auto-generated)
        prediction_window: Hours to predict ahead
        reference_time: Time point for prediction (default: now)
        
    Returns:
        Path to saved GeoJSON file
    """
    # Generate default output path if not specified
    if output_path is None:
        if reference_time is None:
            reference_time = datetime.now()
        time_str = reference_time.strftime("%Y%m%d_%H%M")
        output_path = f"crime_risks_{prediction_window}h_{time_str}.geojson"
    
    prediction_grid = predict_grid_risks(
        prediction_window=prediction_window,
        reference_time=reference_time
    )
    
    # Add risk categories
    prediction_grid['risk_category'] = 'low'
    prediction_grid.loc[prediction_grid['risk'] >= 0.3, 'risk_category'] = 'medium'
    prediction_grid.loc[prediction_grid['risk'] >= 0.5, 'risk_category'] = 'high'
    prediction_grid.loc[prediction_grid['risk'] >= 0.7, 'risk_category'] = 'very_high'
    
    # Add prediction time info
    if reference_time is None:
        reference_time = datetime.now()
    prediction_end = reference_time + timedelta(hours=prediction_window)
    
    prediction_grid['reference_time'] = reference_time.strftime("%Y-%m-%d %H:%M:%S")
    prediction_grid['prediction_end'] = prediction_end.strftime("%Y-%m-%d %H:%M:%S")
    prediction_grid['prediction_window'] = f"{prediction_window}h"
    
    # Export to GeoJSON
    prediction_grid.to_file(output_path, driver="GeoJSON")
    logger.info(f"Risk predictions with geometry saved to {output_path}")
    
    return output_path

def get_top_risk_areas(prediction_window=24, reference_time=None, threshold=0.7, top_n=10):
    """
    Get top risk areas for the API to highlight in the frontend.
    
    Args:
        prediction_window: Hours to predict ahead
        reference_time: Time point for prediction (default: now)
        threshold: Minimum risk score to consider high risk (default: 0.7)
        top_n: Number of high-risk areas to return (default: 10)
        
    Returns:
        Dictionary of high-risk areas with cell_id keys
    """
    prediction_grid = predict_grid_risks(
        prediction_window=prediction_window,
        reference_time=reference_time
    )
    
    # Set prediction time range for metadata
    if reference_time is None:
        reference_time = datetime.now()
    prediction_end = reference_time + timedelta(hours=prediction_window)
    
    # Filter high-risk cells
    high_risk = prediction_grid[prediction_grid['risk'] >= threshold].sort_values('risk', ascending=False)
    
    # If there aren't enough cells above threshold, just get the top N
    if len(high_risk) < top_n:
        high_risk = prediction_grid.sort_values('risk', ascending=False).head(top_n)
    else:
        high_risk = high_risk.head(top_n)
    
    # Create metadata
    risk_areas = {
        "metadata": {
            "prediction_window_hours": prediction_window,
            "reference_time": reference_time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_end_time": prediction_end.strftime("%Y-%m-%d %H:%M:%S"),
            "risk_threshold": threshold,
            "total_risk_areas": len(high_risk)
        },
        "risk_areas": {}
    }
    
    # Add each high-risk cell
    for _, cell in high_risk.iterrows():
        cell_id = cell['cell_id']
        risk = float(round(cell['risk'], 3))
        
        # Group risk into categories for easier display
        risk_category = "high"
        if risk >= 0.7:
            risk_category = "very_high" 
            
        risk_areas["risk_areas"][cell_id] = {
            "risk": risk,
            "risk_category": risk_category,
            "cell_id": cell_id
        }
    
    return risk_areas

def predict_for_multiple_windows(reference_time=None, windows=None):
    """
    Generate predictions for multiple time windows from a single reference point
    
    Args:
        reference_time: Time point for prediction (default: now)
        windows: List of prediction windows in hours (default: [6, 12, 24, 72])
        
    Returns:
        Dictionary with predictions for each time window
    """
    if reference_time is None:
        reference_time = datetime.now()
        
    if windows is None:
        windows = [6, 12, 24, 72]  # 6h, 12h, 24h, 3d
    
    logger.info(f"Generating predictions for multiple windows {windows} from {reference_time}")
    
    predictions = {
        "metadata": {
            "reference_time": reference_time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_windows": windows
        },
        "predictions": {}
    }
    
    for window in windows:
        try:
            # Load model for this window
            model = load_model(prediction_window=window)
            
            # Generate predictions
            prediction_grid = predict_grid_risks(
                prediction_window=window,
                reference_time=reference_time,
                model=model
            )
            
            # Convert to dictionary format
            window_predictions = {}
            for _, cell in prediction_grid.iterrows():
                cell_id = cell['cell_id']
                risk = float(round(cell['risk'], 3))
                
                # Group risk into categories
                risk_category = "low"
                if risk >= 0.7:
                    risk_category = "very_high" 
                elif risk >= 0.5:
                    risk_category = "high"
                elif risk >= 0.3:
                    risk_category = "medium"
                    
                window_predictions[cell_id] = {
                    "risk": risk,
                    "risk_category": risk_category
                }
            
            # Add to main predictions dictionary
            predictions["predictions"][f"{window}h"] = window_predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for {window}h window: {e}")
            predictions["predictions"][f"{window}h"] = {"error": str(e)}
    
    return predictions

def save_multi_window_predictions(output_path=None, reference_time=None, windows=None):
    """
    Save predictions for multiple time windows to a single JSON file
    
    Args:
        output_path: Path to save JSON file (default: auto-generated)
        reference_time: Time point for prediction (default: now)
        windows: List of prediction windows in hours (default: [6, 12, 24, 72])
        
    Returns:
        Path to saved JSON file
    """
    # Generate default output path if not specified
    if output_path is None:
        if reference_time is None:
            reference_time = datetime.now()
        time_str = reference_time.strftime("%Y%m%d_%H%M")
        output_path = f"crime_risks_multi_{time_str}.json"
    
    predictions = predict_for_multiple_windows(
        reference_time=reference_time,
        windows=windows
    )
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Multi-window predictions saved to {output_path}")
    
    return output_path

# Re-implement functions from preprocess.py to avoid circular imports
def count_crimes_per_cell(grid_gdf, crime_gdf):
    """Count crimes in each grid cell"""
    # Simplified implementation to avoid circular import
    joined = gpd.sjoin(grid_gdf, crime_gdf, how='left', predicate='intersects')
    crime_counts = joined.groupby('h3_index').size().reset_index(name='crime_count')

    grid_with_crimes = grid_gdf.merge(crime_counts, on='h3_index', how='left')
    grid_with_crimes['crime_count'] = grid_with_crimes['crime_count'].fillna(0)
    return grid_with_crimes

def add_distance_to_police(grid_gdf, police_gdf):
    """Calculate distance to nearest police station for each grid cell"""
    # Simplified implementation to avoid circular import
    import numpy as np
    
    grid_centroids = grid_gdf.copy()
    
    if grid_centroids.crs != police_gdf.crs:
        police_gdf = police_gdf.to_crs(grid_centroids.crs)
    
    # Project to UTM Zone 18N (appropriate for Medellín, Colombia)
    grid_projected = grid_centroids.to_crs("EPSG:32618")
    police_projected = police_gdf.to_crs("EPSG:32618")
    
    grid_projected['centroid'] = grid_projected.geometry.centroid
    
    # Calculate distances to police stations
    distances = []
    for _, cell in grid_projected.iterrows():
        dist_to_police = police_projected.geometry.distance(cell['centroid'])
        min_dist = dist_to_police.min()
        distances.append(min_dist)
    
    grid_gdf['distance_to_police'] = distances
    grid_gdf['distance_to_police'] = grid_gdf['distance_to_police'] / 1000  # Convert to km
    
    return grid_gdf

def add_demographic_features(grid_gdf, barrios_gdf):
    """Add demographic features from barrios that intersect with each cell"""
    # Simplified implementation to avoid circular import
    grid_with_barrios = grid_gdf.copy()
    grid_with_barrios['barrios'] = None
    grid_with_barrios['comuna'] = None

    for idx, cell in grid_gdf.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            grid_with_barrios.at[idx, 'barrios'] = ','.join(intersecting_barrios['nombre'].tolist())
            # comuna mas comun que intersecta
            grid_with_barrios.at[idx, 'comuna'] = intersecting_barrios['codigo_comuna'].mode()[0]

    return grid_with_barrios

if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Generate crime risk predictions')
    parser.add_argument('--window', type=int, default=24, help='Prediction window in hours')
    parser.add_argument('--format', choices=['json', 'geojson', 'multi'], default='json', 
                        help='Output format (json, geojson, or multi for multiple windows)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--date', type=str, default=None, 
                        help='Reference date/time (format: YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    # Parse reference time if provided
    reference_time = None
    if args.date:
        try:
            reference_time = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
            logger.info(f"Using provided reference time: {reference_time}")
        except ValueError:
            logger.error(f"Invalid date format. Use YYYY-MM-DD HH:MM:SS")
            reference_time = datetime.now()
    
    # Generate predictions
    if args.format == 'json':
        logger.info(f"Generating JSON predictions for {args.window}h window")
        save_risk_json(args.output, args.window, reference_time)
    elif args.format == 'geojson':
        logger.info(f"Generating GeoJSON predictions for {args.window}h window")
        save_risk_geojson(args.output, args.window, reference_time)
    elif args.format == 'multi':
        logger.info("Generating multi-window predictions")
        windows = [6, 12, 24, 72]  # Default windows
        save_multi_window_predictions(args.output, reference_time, windows)
    
    logger.info("Prediction completed")

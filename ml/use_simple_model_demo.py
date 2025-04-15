"""
Simple Demo for Using the Crime Prediction Model
This script shows how to load and use the simplified crime prediction model.
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

#logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path="crime_model_simple.pkl"):
    """Load the trained model"""
    logger.info(f"Loading model from {model_path}")
    try:
        model_data = joblib.load(model_path)
        logger.info(f"Model loaded successfully - trained on {model_data.get('training_date', 'unknown date')}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def prepare_prediction_features(grid_cells):
    """
    Prepare features for making predictions
    
    Args:
        grid_cells: List of dictionaries with h3_index, crime_count, and distance_to_police
    
    Returns:
        DataFrame with features ready for prediction
    """
    logger.info(f"Preparing features for {len(grid_cells)} grid cells")
    
    # Create a DataFrame from the grid cells
    df = pd.DataFrame(grid_cells)
    
    # Add time features for current time
    now = datetime.now()
    df['ref_hour'] = now.hour
    df['ref_day'] = now.weekday()
    df['ref_month'] = now.month
    df['ref_is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    return df

def predict_crime_risk(model_data, grid_cells):
    """
    Generate crime risk predictions
    
    Args:
        model_data: Dictionary containing model, scaler, and metadata
        grid_cells: List of dictionaries with grid cell data
    
    Returns:
        List of dictionaries with grid cell IDs and risk scores
    """
    model = model_data['model']
    scaler = model_data['scaler']
    numerical_columns = model_data['numerical_columns']
    required_features = model_data.get('features', [])
    
    X = prepare_prediction_features(grid_cells)
    
    for feature in required_features:
        if feature not in X.columns:
            logger.warning(f"Missing feature column: {feature} - adding with zeros")
            X[feature] = 0
    
    X = X[required_features]
    
    numeric_cols_present = [col for col in numerical_columns if col in X.columns]
    if numeric_cols_present:
        X[numeric_cols_present] = scaler.transform(X[numeric_cols_present])
    
    # Generate predictions
    try:
        logger.info("Generating predictions...")
        # Get probability of class 1 (crime occurring)
        risk_scores = model.predict_proba(X)[:, 1]
        
        results = []
        for i, cell in enumerate(grid_cells):
            risk_score = float(risk_scores[i])
            
            risk_category = "low"
            if risk_score >= 0.7:
                risk_category = "very_high"
            elif risk_score >= 0.5:
                risk_category = "high"
            elif risk_score >= 0.3:
                risk_category = "medium"
            
            results.append({
                "cell_id": cell.get("h3_index", f"cell-{i+1}"),
                "risk": round(risk_score, 3),
                "risk_category": risk_category
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None

if __name__ == "__main__":
    # Load the model
    model_data = load_model()
    
    if model_data:
        # Generate some example grid cells
        # These are the minimum features needed for prediction:
        # 1. crime_count (historical crime count in the cell)
        # 2. distance_to_police (distance to nearest police station in km)
        # Note: h3_index is only used as an identifier and not used in prediction
        example_cells = [
            {"crime_count": 15, "distance_to_police": 0.8, "h3_index": "cell-001"},
            {"crime_count": 3, "distance_to_police": 1.2, "h3_index": "cell-002"},
            {"crime_count": 0, "distance_to_police": 2.5, "h3_index": "cell-003"},
            {"crime_count": 8, "distance_to_police": 0.5, "h3_index": "cell-004"},
            {"crime_count": 22, "distance_to_police": 3.1, "h3_index": "cell-005"},
        ]
        
        predictions = predict_crime_risk(model_data, example_cells)
        
        if predictions:
            logger.info("Prediction results:")
            for pred in predictions:
                logger.info(f"Cell {pred['cell_id']}: Risk {pred['risk']} ({pred['risk_category']})")
            
            # Format as JSON 
            import json
            
            response = {
                "metadata": {
                    "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_type": "simplified",
                    "total_cells": len(predictions)
                },
                "cells": {pred["cell_id"]: pred for pred in predictions}
            }
            
            print("\nExample JSON for API response:")
            print(json.dumps(response, indent=2))
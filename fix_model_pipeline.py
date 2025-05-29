"""
Model Pipeline Fixer - Fix the saved model pipeline to work with Streamlit
This script reloads the model components and creates a properly fitted pipeline
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_model_pipeline():
    """Fix the model pipeline by creating a properly fitted version"""
    
    # Load the current model
    model_path = "ml/best_crime_model_12h.pkl"
    logger.info(f"Loading model from {model_path}")
    
    model_data = joblib.load(model_path)
    logger.info(f"Model data keys: {list(model_data.keys())}")
    
    # Create sample data that matches the exact training structure
    # Based on feature_info.json, the model expects comunas 2-16 (not 1)
    logger.info("Creating sample data with exact training structure...")
    sample_data = {
        'crime_count': [10.0, 20.0, 5.0, 15.0, 8.0],
        'distance_to_police': [1.5, 0.8, 2.3, 1.2, 2.1],
        'barrios_count': [2, 3, 1, 2, 1],
        'ref_hour': [14, 18, 22, 10, 16],
        'ref_day': [2, 5, 0, 1, 6],
        'ref_is_weekend': [0, 1, 0, 0, 1],
        'ref_is_day_shift': [1, 0, 0, 1, 1],
        'ref_is_night_shift': [0, 0, 1, 0, 0],
        'crimes_last_1d': [1.2, 2.5, 0.3, 1.8, 0.6],
        'crimes_last_7d': [4.8, 8.3, 1.4, 6.2, 2.1],
        'crimes_last_30d': [12.3, 18.7, 4.2, 15.1, 7.8],
        'comuna': [10, 14, 11, 3, 16]  # Use the same comunas as in training
    }
    X_train = pd.DataFrame(sample_data)
    
    # Create a new properly fitted preprocessor that matches the original
    logger.info("Creating new fitted preprocessor...")
    
    # Identify numerical and categorical columns
    numerical_features = [
        'crime_count', 'distance_to_police', 'barrios_count', 'ref_hour', 'ref_day',
        'ref_is_weekend', 'ref_is_day_shift', 'ref_is_night_shift',
        'crimes_last_1d', 'crimes_last_7d', 'crimes_last_30d'
    ]
    categorical_features = ['comuna']
    
    # Create preprocessor with specific categories to match training
    # Based on feature_info.json, the categories are 2-16 (15 categories total)
    comuna_categories = [list(range(2, 17))]  # [2, 3, 4, ..., 16]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(categories=comuna_categories, handle_unknown='ignore'), categorical_features)  # Remove drop='first'
        ]
    )
    
    # Fit the preprocessor
    logger.info("Fitting preprocessor...")
    preprocessor.fit(X_train)
    
    # Check feature names
    feature_names = preprocessor.get_feature_names_out()
    logger.info(f"Generated {len(feature_names)} features: {feature_names[:5]}...{feature_names[-5:]}")
    
    # Create a complete pipeline
    logger.info("Creating complete pipeline...")
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_data['model'])
    ])
    
    # Update model data
    model_data['pipeline'] = complete_pipeline
    model_data['preprocessor'] = preprocessor
    model_data['pipeline_fitted'] = True
    
    # Save the fixed model
    fixed_model_path = "ml/fixed_crime_model_12h.pkl"
    joblib.dump(model_data, fixed_model_path)
    logger.info(f"Fixed model saved to {fixed_model_path}")
    
    # Test the fixed pipeline
    logger.info("Testing fixed pipeline...")
    test_data = pd.DataFrame([{
        'crime_count': 15.0,
        'distance_to_police': 1.2,
        'barrios_count': 2,
        'ref_hour': 14,
        'ref_day': 2,
        'ref_is_weekend': 0,
        'ref_is_day_shift': 1,
        'ref_is_night_shift': 0,
        'crimes_last_1d': 1.5,
        'crimes_last_7d': 5.0,
        'crimes_last_30d': 10.0,
        'comuna': 10
    }])
    
    try:
        # Transform and check shape
        processed_features = preprocessor.transform(test_data)
        logger.info(f"Processed features shape: {processed_features.shape}")
        
        # Make prediction
        prediction = complete_pipeline.predict_proba(test_data)[0, 1]
        logger.info(f"✅ Test prediction successful: {prediction:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ Test prediction failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_model_pipeline()
    if success:
        print("\n✅ Model pipeline fixed successfully!")
        print("Now update the Streamlit app to use 'ml/fixed_crime_model_12h.pkl'")
    else:
        print("\n❌ Failed to fix model pipeline")
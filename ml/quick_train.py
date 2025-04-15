"""
Quick Training Script for Crime Prediction Model
This script trains a simple model for crime prediction with minimal preprocessing.
It's designed for rapid development and integration testing.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse
import logging
import os
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quick_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def quick_preprocess(data_dir='../geodata', max_windows=10, sample_size=10000, 
                    prediction_window=24, random_state=42):
    """
    Perform a simplified preprocessing for quick model training
    
    Args:
        data_dir: Directory containing geodata files
        max_windows: Maximum number of time windows to process (smaller = faster)
        sample_size: Size of data sample to use (smaller = faster)
        prediction_window: Prediction window in hours
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with prepared features and target
    """
    logger.info(f"Quick preprocessing started with max_windows={max_windows}, sample_size={sample_size}")
    
    try:
        # Make sure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Required GitHub files to download if not found locally
        required_files = [
            {"filename": "hex_grid.gpkg", "local_path": os.path.join(data_dir, "hex_grid.gpkg")},
            {"filename": "crime_data1.geojson", "local_path": os.path.join(data_dir, "crime_data1.geojson")},
            {"filename": "police1.geojson", "local_path": os.path.join(data_dir, "police1.geojson")},
            {"filename": "barrios_medellin1.geojson", "local_path": os.path.join(data_dir, "barrios_medellin1.geojson")}
        ]
        
        # Load the grid
        grid_file = os.path.join(data_dir, "hex_grid.gpkg")
        if not os.path.exists(grid_file):
            logger.info(f"Grid file not found at {grid_file}, attempting to download...")
            success = download_github_file("hex_grid.gpkg", grid_file)
            if not success:
                raise FileNotFoundError(f"Failed to download grid file from GitHub")
        
        grid_gdf = gpd.read_file(grid_file)
        logger.info(f"Loaded grid with {len(grid_gdf)} cells")
        
        # Load crime data - first try crime_data1.geojson
        crime_file = os.path.join(data_dir, 'crime_data1.geojson')
        if not os.path.exists(crime_file):
            logger.info(f"Crime data file not found at {crime_file}, attempting to download...")
            success = download_github_file("crime_data1.geojson", crime_file)
            if not success:
                # Try alternatives
                crime_file = os.path.find_in_directory(data_dir, r'hurto_a_persona\.csv')
                if not crime_file:
                    crime_file = next(
                        (f for f in [os.path.join('..', 'ml', 'data', 'hurto_a_persona.csv'), 
                                    os.path.join('..', 'ml', 'data', 'data.csv')] 
                        if os.path.exists(f)), 
                        None
                    )
                
                if not crime_file:
                    raise FileNotFoundError("Crime data file not found and failed to download")
        
        # Determine if file is CSV or GeoJSON
        if crime_file.endswith('.csv'):
            crime_data = pd.read_csv(crime_file)
            # We'll need to convert to GeoDataFrame later
        else:
            # It's a GeoJSON
            crime_data = gpd.read_file(crime_file)
            
        logger.info(f"Loaded crime data with {len(crime_data)} records")
        
        # Load police data
        police_file = os.path.join(data_dir, 'police1.geojson')
        if not os.path.exists(police_file):
            logger.info(f"Police station file not found at {police_file}, attempting to download...")
            success = download_github_file("police1.geojson", police_file)
            if not success:
                # Try alternative
                police_file = os.path.join(data_dir, 'police.geojson')
                if not os.path.exists(police_file):
                    success = download_github_file("police.geojson")
                    if not success:
                        logger.warning("Police data not found and failed to download, will not use distance to police feature")
                        police_gdf = None
                    else:
                        police_gdf = gpd.read_file(police_file)
                else:
                    police_gdf = gpd.read_file(police_file)
            else:
                police_gdf = gpd.read_file(police_file)
        else:
            police_gdf = gpd.read_file(police_file)

        # Convert crime data to GeoDataFrame if it's a DataFrame
        if not isinstance(crime_data, gpd.GeoDataFrame):
            if 'geometry' not in crime_data.columns:
                if all(col in crime_data.columns for col in ['latitud', 'longitud']):
                    from shapely.geometry import Point
                    crime_data['geometry'] = [
                        Point(lon, lat) for lon, lat in zip(crime_data['longitud'], crime_data['latitud'])
                    ]
                elif all(col in crime_data.columns for col in ['latitude', 'longitude']):
                    from shapely.geometry import Point
                    crime_data['geometry'] = [
                        Point(lon, lat) for lon, lat in zip(crime_data['longitude'], crime_data['latitude'])
                    ]
                else:
                    raise ValueError("Crime data must have coordinate columns (latitud/longitud or latitude/longitude)")
            
            crime_gdf = gpd.GeoDataFrame(crime_data, geometry='geometry', crs="EPSG:4326")
        else:
            crime_gdf = crime_data
            
        # Ensure crime_gdf has a CRS
        if crime_gdf.crs is None:
            crime_gdf.crs = "EPSG:4326"
        
        # Ensure consistent CRS
        if police_gdf is not None and police_gdf.crs != "EPSG:4326":
            police_gdf = police_gdf.to_crs("EPSG:4326")
        
        if grid_gdf.crs != "EPSG:4326":
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
        
        # Join crime data to grid
        grid_gdf['crime_count'] = 0
        crime_in_grid = gpd.sjoin(crime_gdf, grid_gdf, how='inner', predicate='within')
        crime_counts = crime_in_grid.groupby('index_right').size()
        grid_gdf.loc[crime_counts.index, 'crime_count'] = crime_counts.values
        
        # Add distance to nearest police station
        if police_gdf is not None:
            logger.info("Calculating distance to nearest police station...")
            
            # Project to UTM for distance calculations
            grid_utm = grid_gdf.copy().to_crs(epsg=32618)  # UTM Zone 18N
            police_utm = police_gdf.copy().to_crs(epsg=32618)
            
            # Calculate centroids if not points
            if grid_utm.geometry.geom_type[0] != 'Point':
                grid_centers = grid_utm.copy()
                grid_centers.geometry = grid_utm.geometry.centroid
            else:
                grid_centers = grid_utm
            
            # Calculate distances
            from shapely.geometry import Point
            from scipy.spatial import cKDTree
            
            police_coords = np.array([
                (geom.x, geom.y) for geom in police_utm.geometry
            ])
            
            grid_coords = np.array([
                (geom.x, geom.y) for geom in grid_centers.geometry
            ])
            
            # Build KD-tree for police stations
            police_tree = cKDTree(police_coords)
            
            # Query tree for each grid centroid
            distances, _ = police_tree.query(grid_coords)
            
            # Convert from m to km
            grid_gdf['distance_to_police'] = distances / 1000
            logger.info(f"Distance to police calculated. Range: {grid_gdf['distance_to_police'].min():.2f} - {grid_gdf['distance_to_police'].max():.2f} km")
        else:
            # Use random values for distance if police data unavailable
            grid_gdf['distance_to_police'] = np.random.uniform(0.1, 5.0, size=len(grid_gdf))
            logger.warning("Using random values for distance to police")
        
        # Prepare dataset with time features
        logger.info("Creating time windows for crime prediction...")
        
        # Parse date if it's a string
        if 'fecha' in crime_data.columns:
            date_col = 'fecha'
        elif 'date' in crime_data.columns:
            date_col = 'date'
        elif 'fecha_hecho' in crime_data.columns:
            date_col = 'fecha_hecho'
        else:
            date_col = next((col for col in crime_data.columns if 'date' in col.lower() or 'fecha' in col.lower()), None)
        
        if date_col:
            if isinstance(crime_data[date_col].iloc[0], str):
                crime_data['datetime'] = pd.to_datetime(crime_data[date_col], errors='coerce')
            else:
                crime_data['datetime'] = crime_data[date_col]
        else:
            # Generate random dates if no date column
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 12, 31)
            days_range = (end_date - start_date).days
            random_days = np.random.randint(0, days_range, size=len(crime_data))
            random_hours = np.random.randint(0, 24, size=len(crime_data))
            crime_data['datetime'] = [
                start_date + timedelta(days=d, hours=h)
                for d, h in zip(random_days, random_hours)
            ]
            logger.warning("No date column found, using randomly generated dates")
        
        # Generate time windows
        all_cells = grid_gdf['h3_index'].unique()
        time_periods = []
        
        # Create a few reference timepoints
        start_date = crime_data['datetime'].min() + timedelta(days=7)  # Skip first week
        end_date = crime_data['datetime'].max() - timedelta(days=7)    # Skip last week
        
        # Generate evenly spaced timepoints
        timepoints = pd.date_range(
            start=start_date, 
            end=end_date, 
            periods=min(max_windows, 50)  # Limit number of windows
        )
        
        logger.info(f"Generated {len(timepoints)} reference time points from {timepoints[0]} to {timepoints[-1]}")
        
        # Process each reference timepoint
        for i, ref_time in enumerate(timepoints):
            if i > max_windows:
                break
                
            logger.info(f"Processing window {i+1}/{len(timepoints)} at {ref_time}")
            
            # Define the target window
            target_end = ref_time + timedelta(hours=prediction_window)
            
            # Find crimes in the target window
            target_crimes = crime_gdf[
                (crime_data['datetime'] >= ref_time) & 
                (crime_data['datetime'] < target_end)
            ]
            
            # Join to grid to count target crimes per cell
            if len(target_crimes) > 0:
                target_in_grid = gpd.sjoin(target_crimes, grid_gdf, how='inner', predicate='within')
                target_counts = target_in_grid.groupby('index_right').size()
                
                # Create a map of cell index to crime count
                target_map = {idx: count for idx, count in zip(target_counts.index, target_counts.values)}
            else:
                target_map = {}
            
            # Create features for each cell
            for cell_idx, cell in enumerate(all_cells):
                if cell_idx >= sample_size / len(timepoints):
                    # Limit cells per timepoint to control total sample size
                    break
                    
                # Get grid cell row
                cell_row = grid_gdf[grid_gdf['h3_index'] == cell].iloc[0]
                
                # Basic features
                features = {
                    'h3_index': cell,
                    'reference_time': ref_time,
                    'ref_hour': ref_time.hour,
                    'ref_day': ref_time.weekday(),
                    'ref_month': ref_time.month,
                    'ref_is_weekend': 1 if ref_time.weekday() >= 5 else 0,
                    'crime_count': cell_row['crime_count'],
                    'distance_to_police': cell_row['distance_to_police']
                }
                
                # Target value (1 if any crimes in window, 0 otherwise)
                grid_idx = cell_row.name
                target_value = 1 if grid_idx in target_map and target_map[grid_idx] > 0 else 0
                features[f'target_{prediction_window}h'] = target_value
                
                time_periods.append(features)
        
        # Convert to DataFrame
        temporal_df = pd.DataFrame(time_periods)
        logger.info(f"Created dataset with {len(temporal_df)} samples")
        
        # Balance the dataset if needed
        positive_samples = temporal_df[temporal_df[f'target_{prediction_window}h'] == 1]
        negative_samples = temporal_df[temporal_df[f'target_{prediction_window}h'] == 0]
        
        positive_count = len(positive_samples)
        negative_count = len(negative_samples)
        
        # If imbalanced, undersample majority class
        if positive_count < negative_count / 3:
            # Severe imbalance, undersample negatives
            logger.info(f"Balancing dataset: {positive_count} positive, {negative_count} negative samples")
            undersampled_negatives = negative_samples.sample(
                min(positive_count * 3, negative_count),
                random_state=random_state
            )
            temporal_df = pd.concat([positive_samples, undersampled_negatives])
            logger.info(f"Balanced dataset: {len(temporal_df)} samples")
        
        return temporal_df
        
    except Exception as e:
        logger.exception(f"Error during quick preprocessing: {e}")
        return None

def find_in_directory(dir_path, filename_pattern):
    """Find a file matching pattern in directory"""
    import re
    pattern = re.compile(filename_pattern)
    for file in os.listdir(dir_path):
        if pattern.search(file):
            return os.path.join(dir_path, file)
    return None

# Add this method to os.path module to use in quick_preprocess
os.path.find_in_directory = find_in_directory

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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file
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

def train_quick_model(dataset, target_col='target_24h', model_file=None, test_size=0.25):
    """
    Train a quick random forest model for crime prediction
    
    Args:
        dataset: DataFrame with features and target
        target_col: Name of the target column
        model_file: File to save the model
        test_size: Fraction of data to use for testing
    
    Returns:
        Dictionary with model and metrics
    """
    logger.info(f"Training quick model for {target_col}")
    
    # Ensure target column exists
    if target_col not in dataset.columns:
        alternatives = [col for col in dataset.columns if col.startswith('target_')]
        if not alternatives:
            logger.error(f"No target column found in dataset (looking for {target_col})")
            return None
        
        target_col = alternatives[0]
        logger.warning(f"Target column {target_col} not found, using {target_col} instead")
    
    # Split into features and target
    drop_cols = ['h3_index', 'geometry', 'reference_time'] if 'geometry' in dataset.columns else ['h3_index', 'reference_time']
    drop_cols += [col for col in dataset.columns if col.startswith('target_') and col != target_col]
    
    X = dataset.drop(drop_cols, axis=1, errors='ignore')
    y = dataset[target_col]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    logger.info(f"Positive samples in training: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train):.1%})")
    
    # Identify numerical columns for scaling
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train a random forest classifier
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,  # Use fewer trees for speed
        min_samples_leaf=10,  # Avoid overfitting
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
        logger.info(f"ROC AUC: {auc:.4f}")
    except:
        auc = None
        logger.warning("Could not calculate AUC")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Save model - SIMPLIFIED: Only save one file
    if model_file is None:
        model_file = "crime_model_simple.pkl"
        
    # Create a dictionary with model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'numerical_columns': numerical_cols,
        'features': X_train.columns.tolist(),
        'target': target_col,
        'prediction_window': int(target_col.split('_')[-1].replace('h', '')),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc': auc
        }
    }
    
    logger.info(f"Saving model to {model_file}")
    joblib.dump(model_data, model_file)
    
    return model_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a quick crime prediction model')
    parser.add_argument('--data_dir', type=str, default='../geodata', help='Directory with geodata')
    parser.add_argument('--windows', type=int, default=10, help='Number of time windows to process')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--window', type=int, default=24, help='Prediction window in hours')
    parser.add_argument('--output', type=str, default=None, help='Output model file')
    
    args = parser.parse_args()
    
    logger.info(f"Quick training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Process data
        dataset = quick_preprocess(
            data_dir=args.data_dir,
            max_windows=args.windows,
            sample_size=args.samples,
            prediction_window=args.window
        )
        
        if dataset is None or len(dataset) == 0:
            logger.error("Failed to create dataset")
            exit(1)
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Save a sample for inspection
        sample_file = f"crime_dataset_quick_sample_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        dataset.sample(min(1000, len(dataset))).to_csv(sample_file, index=False)
        logger.info(f"Sample saved to {sample_file}")
        
        # Train model
        target_col = f"target_{args.window}h"
        model_data = train_quick_model(dataset, target_col, args.output)
        
        if model_data is None:
            logger.error("Failed to train model")
            exit(1)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
    
    logger.info(f"Process finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
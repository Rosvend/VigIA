"""
Preprocessing script for crime prediction dataset with 12-hour windows
This script processes the crime data, police station locations, and grid cells to create a dataset suitable for machine learning models.
Focus on 12-hour temporal windows for better class balance and temporal resolution 8 H3 hexagons.
"""

# Import necessary libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import warnings
import requests
import logging
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def download_github_file(filename, output_path, base_url="https://github.com/Rosvend/Patrol-routes-optimization-Medellin/raw/main/geodata/"):
    """Downloads a file from a GitHub repository and saves it to output_path."""
    url = base_url + filename
    log.info(f"Downloading {filename} from GitHub...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            log.info(f"Successfully downloaded {filename} to {output_path}")
            return True
        else:
            log.error(f"Failed to download {filename}. Status code: {response.status_code}")
            return False
    except Exception as e:
        log.error(f"Error downloading {filename}: {e}")
        return False

def load_datasets(data_dir="../geodata", resolution=8):
    """
    Downloads all datasets from GitHub and loads them into GeoDataFrames.
    Prioritizes H3 resolution 8 grid for better balance between granularity and data density.
    
    Args:
        data_dir: Directory to save/load data files
        resolution: H3 resolution to use (default 8)
    
    Returns:
        grid_gdf: H3 grid cells (resolution 8)
        gdf_crimenes: Crime data
        gdf_police: Police station locations
        gdf_barrios: Neighborhoods data
        gdf_turismo: Tourist attractions data
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, data_dir))
    os.makedirs(data_dir, exist_ok=True)
    log.info(f"Using data directory: {data_dir}")
    log.info(f"Target H3 resolution: {resolution}")

    # Define files with priority for resolution 8
    files = {
        f"hex_grid_res{resolution}.gpkg": "grid_gdf",
        "crime_data1.geojson": "gdf_crimenes",
        "police1.geojson": "gdf_police",
        "barrios_medellin1.geojson": "gdf_barrios",
        "atractivos_turisticos1.geojson": "gdf_turismo"
    }

    loaded_data = {}
    for filename, varname in files.items():
        path = os.path.join(data_dir, filename)
        
        # Check if file exists locally first
        if not os.path.exists(path):
            success = download_github_file(filename, path)
            if not success and varname == "grid_gdf":
                # Fallback to generic grid file if specific resolution not available
                fallback_file = "hex_grid.gpkg"
                fallback_path = os.path.join(data_dir, fallback_file)
                log.warning(f"Trying fallback grid file: {fallback_file}")
                success = download_github_file(fallback_file, fallback_path)
                if success:
                    path = fallback_path
                else:
                    raise FileNotFoundError(f"Could not download grid file for resolution {resolution}")
        
        try:
            gdf = gpd.read_file(path)
            loaded_data[varname] = gdf
            
            if varname == "grid_gdf":
                log.info(f"Loaded grid with {len(gdf)} cells")
                # Check if this is actually resolution 8
                if 'h3_index' in gdf.columns and len(gdf) > 0:
                    # H3 resolution 8 should have around 1000-3000 cells for Medellín
                    if 500 < len(gdf) < 5000:
                        log.info(f"Grid appears to be appropriate resolution (H3 res ~{resolution})")
                    else:
                        log.warning(f"Grid size ({len(gdf)} cells) may not match expected H3 resolution {resolution}")
            else:
                log.info(f"Loaded {filename} with {len(gdf)} records")
                
        except Exception as e:
            log.error(f"Error loading {filename}: {e}")
            raise

    return (loaded_data["grid_gdf"],
            loaded_data["gdf_crimenes"],
            loaded_data["gdf_police"],
            loaded_data["gdf_barrios"],
            loaded_data["gdf_turismo"])

def crs_harmonization(grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, gdf_turismo):
    """Harmonize the coordinate reference systems (CRS) of all datasets to EPSG:4326."""
    log.info("Checking and harmonizing coordinate reference systems (CRS)...")

    def ensure_crs(gdf, name):
        if gdf.crs is None:
            log.info(f"Setting {name} CRS to EPSG:4326")
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs != "EPSG:4326":
            log.info(f"Converting {name} CRS from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    grid_gdf     = ensure_crs(grid_gdf, "grid")
    gdf_crimenes = ensure_crs(gdf_crimenes, "crime data")
    gdf_police   = ensure_crs(gdf_police, "police data")
    gdf_barrios  = ensure_crs(gdf_barrios, "barrios data")
    gdf_turismo  = ensure_crs(gdf_turismo, "tourist data")

    log.info("All datasets now have matching CRS: EPSG:4326")
    return grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, gdf_turismo

def count_crimes_per_cell(grid_gdf, crime_gdf):
    """Count crimes in each grid cell"""
    log.info("Counting total crimes per grid cell...")
    joined = gpd.sjoin(grid_gdf, crime_gdf, how='left', predicate='intersects')
    crime_counts = joined.groupby('h3_index').size().reset_index(name='crime_count')

    grid_with_crimes = grid_gdf.merge(crime_counts, on='h3_index', how='left')
    grid_with_crimes['crime_count'] = grid_with_crimes['crime_count'].fillna(0)
    
    log.info(f"Crime counts range: {grid_with_crimes['crime_count'].min()} - {grid_with_crimes['crime_count'].max()}")
    log.info(f"Cells with crimes: {(grid_with_crimes['crime_count'] > 0).sum()}/{len(grid_with_crimes)}")
    return grid_with_crimes

def add_distance_to_police(grid_gdf, police_gdf):
    """Calculate distance to nearest police station for each grid cell"""
    log.info("Calculating distance to nearest police station...")
    grid_centroids = grid_gdf.copy()
    
    if grid_centroids.crs != police_gdf.crs:
        log.info(f"Converting police stations CRS to match grid CRS: {grid_centroids.crs}")
        police_gdf = police_gdf.to_crs(grid_centroids.crs)
    
    # Project to UTM Zone 18N for accurate distance calculations (Colombia)
    log.info("Projecting to UTM Zone 18N (EPSG:32618) for accurate distance calculations")
    grid_projected = grid_centroids.to_crs("EPSG:32618")
    police_projected = police_gdf.to_crs("EPSG:32618")
    
    grid_projected['centroid'] = grid_projected.geometry.centroid
    
    # Calculate distances using vectorized operations for better performance
    from scipy.spatial import cKDTree
    
    # Extract coordinates
    police_coords = np.array([
        (geom.x, geom.y) for geom in police_projected.geometry
    ])
    
    grid_coords = np.array([
        (geom.x, geom.y) for geom in grid_projected['centroid']
    ])
    
    # Build KD-tree for police stations
    police_tree = cKDTree(police_coords)
    
    # Query tree for each grid centroid
    distances, _ = police_tree.query(grid_coords)
    
    # Convert from meters to kilometers
    grid_gdf['distance_to_police'] = distances / 1000
    
    log.info(f"Distance calculations complete. Range: {grid_gdf['distance_to_police'].min():.2f} - {grid_gdf['distance_to_police'].max():.2f} km")
    return grid_gdf

def generate_temporal_windows_12h(grid_gdf, crime_gdf, prediction_window=12, step_size=12, 
                                  quick_mode=False, max_windows=None, sample_size=100):
    """
    Generate temporal sliding windows focused on 12-hour predictions
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        crime_gdf: GeoDataFrame with crime data
        prediction_window: Hours for prediction target (default: 12)
        step_size: Hours to step forward for each new training window (default: 12)
        quick_mode: If True, process fewer windows and cells for faster execution
        max_windows: Maximum number of windows to process
        sample_size: Number of grid cells to sample in quick mode
                   
    Returns:
        DataFrame with temporal windows focused on 12-hour predictions
    """
    log.info(f"Generating 12-hour temporal windows (prediction: {prediction_window}h, step: {step_size}h)")
    
    # Get date range from crime data
    start_date = crime_gdf['fecha_hecho'].min()
    end_date = crime_gdf['fecha_hecho'].max()
    
    # Allow for feature generation lookback
    feature_lookback = timedelta(days=30 if quick_mode else 90)
    training_start = start_date + feature_lookback
    
    log.info(f"Data spans from {start_date} to {end_date}")
    log.info(f"Training windows will start from {training_start}")
    
    # Generate reference points every 12 hours
    reference_points = []
    current = training_start
    while current < end_date - timedelta(hours=prediction_window):
        reference_points.append(current)
        current += timedelta(hours=step_size)
    
    # Limit windows in quick mode
    if quick_mode and max_windows:
        if len(reference_points) > max_windows:
            indices = np.linspace(0, len(reference_points) - 1, max_windows, dtype=int)
            reference_points = [reference_points[i] for i in indices]
        log.info(f"QUICK MODE: Using {len(reference_points)} windows")
    
    # Sample grid cells in quick mode
    if quick_mode:
        actual_sample = min(sample_size, len(grid_gdf))
        log.info(f"QUICK MODE: Using {actual_sample} grid cells")
        grid_gdf = grid_gdf.sample(actual_sample, random_state=42)
    
    log.info(f"Processing {len(reference_points)} reference time points")
    
    # Build spatial index for performance
    crime_sindex = crime_gdf.sindex
    
    window_data = []
    recency_windows = [1, 7, 30]  # Days
    
    # Process each reference point
    for i, ref_point in enumerate(reference_points):
        if i % 10 == 0:
            log.info(f"Processing window {i+1}/{len(reference_points)} at {ref_point}")
        
        # Create window dataframe
        window_gdf = grid_gdf.copy()
        
        # Convert to naive datetime for consistency
        naive_ref_point = ref_point.replace(tzinfo=None) if ref_point.tzinfo else ref_point
        window_gdf['reference_time'] = naive_ref_point
        
        # Add temporal features
        window_gdf['ref_hour'] = ref_point.hour
        window_gdf['ref_day'] = ref_point.dayofweek
        window_gdf['ref_month'] = ref_point.month
        window_gdf['ref_is_weekend'] = 1 if ref_point.dayofweek >= 5 else 0
        window_gdf['ref_is_day_shift'] = 1 if 6 <= ref_point.hour < 18 else 0  # 6 AM - 6 PM
        window_gdf['ref_is_night_shift'] = 1 if ref_point.hour < 6 or ref_point.hour >= 18 else 0
        
        # Add recent crime features
        for window_days in recency_windows:
            window_gdf[f'crimes_last_{window_days}d'] = 0
            cutoff = ref_point - timedelta(days=window_days)
            recent_crimes = crime_gdf[
                (crime_gdf['fecha_hecho'] >= cutoff) & 
                (crime_gdf['fecha_hecho'] < ref_point)
            ]
            
            if len(recent_crimes) > 0:
                # Count crimes in each cell for this time window
                for idx, cell in window_gdf.iterrows():
                    try:
                        # Use spatial intersection
                        crimes_in_cell = recent_crimes[recent_crimes.intersects(cell.geometry)]
                        if len(crimes_in_cell) > 0:
                            # Apply recency weighting
                            days_from_cutoff = (crimes_in_cell['fecha_hecho'] - cutoff).dt.total_seconds() / (24 * 3600)
                            weights = np.exp(days_from_cutoff / window_days)
                            window_gdf.at[idx, f'crimes_last_{window_days}d'] = weights.sum()
                    except Exception as e:
                        if not quick_mode:
                            log.debug(f"Warning processing cell {idx}: {e}")
                        continue
        
        # Generate 12-hour target
        target_col = f'target_{prediction_window}h'
        window_gdf[target_col] = 0
        
        future_cutoff = ref_point + timedelta(hours=prediction_window)
        future_crimes = crime_gdf[
            (crime_gdf['fecha_hecho'] >= ref_point) & 
            (crime_gdf['fecha_hecho'] < future_cutoff)
        ]
        
        if len(future_crimes) > 0:
            for idx, cell in window_gdf.iterrows():
                try:
                    # Check if any future crimes intersect with this cell
                    has_crime = any(future_crimes.intersects(cell.geometry))
                    if has_crime:
                        window_gdf.at[idx, target_col] = 1
                except Exception as e:
                    if not quick_mode:
                        log.debug(f"Warning processing future crimes for cell {idx}: {e}")
                    continue
        
        window_data.append(window_gdf)
    
    # Combine all windows
    if not window_data:
        log.error("No temporal windows generated")
        return None
        
    combined_windows = pd.concat(window_data, ignore_index=True)
    log.info(f"Combined temporal dataset shape: {combined_windows.shape}")
    
    # Report class distribution
    target_col = f'target_{prediction_window}h'
    if target_col in combined_windows.columns:
        positive_count = combined_windows[target_col].sum()
        negative_count = len(combined_windows) - positive_count
        total = len(combined_windows)
        
        log.info(f"\n{prediction_window}-hour target class distribution:")
        log.info(f"  No crime (0): {negative_count} ({negative_count/total:.2%})")
        log.info(f"  Crime (1): {positive_count} ({positive_count/total:.2%})")
        log.info(f"  Positive class ratio: {positive_count/total:.3%}")
        
        # Check if balance is better than the original 6-hour windows
        if positive_count/total > 0.02:  # More than 2%
            log.info(f"✓ Good class balance achieved with 12-hour windows!")
        else:
            log.warning(f"Class balance still low. Consider longer windows or different approach.")
    
    return combined_windows

def create_temporal_splits(dataset, target_col='target_12h'):
    """
    Create temporal train/validation/test splits following the requirements:
    - Train: January to September (70%)
    - Validation: October to November (15%) 
    - Test: December (15%)
    """
    log.info("Creating temporal train/validation/test splits...")
    
    # Ensure reference_time is datetime
    dataset['reference_time'] = pd.to_datetime(dataset['reference_time'])
    dataset['ref_month'] = dataset['reference_time'].dt.month
    
    # Create splits based on months
    train_data = dataset[dataset['ref_month'] <= 9].copy()  # Jan-Sep
    val_data = dataset[(dataset['ref_month'] > 9) & (dataset['ref_month'] <= 11)].copy()  # Oct-Nov
    test_data = dataset[dataset['ref_month'] > 11].copy()  # Dec
    
    log.info(f"Temporal splits created:")
    log.info(f"  Training (Jan-Sep): {len(train_data)} samples")
    log.info(f"  Validation (Oct-Nov): {len(val_data)} samples")
    log.info(f"  Test (Dec): {len(test_data)} samples")
    
    # Check class distributions
    for name, data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        if len(data) > 0 and target_col in data.columns:
            pos_count = data[target_col].sum()
            total = len(data)
            log.info(f"  {name} - Positive class: {pos_count}/{total} ({pos_count/total:.2%})")
    
    return train_data, val_data, test_data

def create_crime_prediction_dataset_12h(grid_gdf, crime_gdf, police_gdf, barrios_gdf, 
                                        prediction_window=12, quick_mode=False):
    """
    Create a complete crime prediction dataset focused on 12-hour windows
    
    Args:
        grid_gdf: GeoDataFrame with H3 resolution 8 grid cells
        crime_gdf: GeoDataFrame with crime data
        police_gdf: GeoDataFrame with police station locations
        barrios_gdf: GeoDataFrame with neighborhood data
        prediction_window: Hours for prediction target (default: 12)
        quick_mode: If True, process fewer data points for faster execution
    
    Returns:
        DataFrame with 12-hour temporal windows for crime prediction
    """
    log.info(f"\n=== Creating 12-hour crime prediction dataset ===")
    log.info(f"Target prediction window: {prediction_window} hours")
    
    log.info("\n1. Counting crimes per grid cell...")
    grid_with_crimes = count_crimes_per_cell(grid_gdf, crime_gdf)
    
    log.info("\n2. Calculating distance to nearest police station...")
    grid_with_police = add_distance_to_police(grid_with_crimes, police_gdf)
    
    log.info("\n3. Adding demographic features...")
    # Simple demographic features from barrios
    grid_with_demographics = grid_with_police.copy()
    grid_with_demographics['barrios_count'] = 0
    grid_with_demographics['comuna'] = None
    
    for idx, cell in grid_with_demographics.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            grid_with_demographics.at[idx, 'barrios_count'] = len(intersecting_barrios)
            # Use most common comuna
            if 'codigo_comuna' in intersecting_barrios.columns:
                comuna_mode = intersecting_barrios['codigo_comuna'].mode()
                if len(comuna_mode) > 0:
                    grid_with_demographics.at[idx, 'comuna'] = comuna_mode.iloc[0]
    
    log.info("\n4. Generating 12-hour temporal windows...")
    temporal_dataset = generate_temporal_windows_12h(
        grid_with_demographics,
        crime_gdf,
        prediction_window=prediction_window,
        step_size=12,  # 12-hour steps
        quick_mode=quick_mode,
        max_windows=20 if quick_mode else None,
        sample_size=100 if quick_mode else None
    )
    
    if temporal_dataset is None:
        log.error("Failed to generate temporal dataset")
        return None
    
    log.info(f"\n=== 12-hour crime prediction dataset complete ===")
    log.info(f"Final dataset shape: {temporal_dataset.shape}")
    
    # Create and display temporal splits
    train_data, val_data, test_data = create_temporal_splits(temporal_dataset, f'target_{prediction_window}h')
    
    return temporal_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess crime data for 12-hour prediction windows')
    parser.add_argument('--quick', action='store_true', help='Process a smaller subset for faster execution')
    parser.add_argument('--resolution', type=int, default=8, help='H3 resolution for grid (default: 8)')
    parser.add_argument('--window', type=int, default=12, help='Prediction window in hours (default: 12)')
    args = parser.parse_args()
    
    log.info(f"Preprocessing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Configuration: H3 resolution {args.resolution}, {args.window}h prediction window")
    
    if args.quick:
        log.info("\n===== QUICK MODE: Processing reduced dataset for rapid testing =====\n")
    
    try:
        # Load datasets with specified resolution
        grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, gdf_turismo = load_datasets(resolution=args.resolution)
        
        # Harmonize coordinate systems
        grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, gdf_turismo = crs_harmonization(
            grid_gdf, gdf_crimenes, gdf_police, gdf_barrios, gdf_turismo
        )
        
        # Create the 12-hour focused dataset
        crime_prediction_dataset = create_crime_prediction_dataset_12h(
            grid_gdf,
            gdf_crimenes,
            gdf_police,
            gdf_barrios,
            prediction_window=args.window,
            quick_mode=args.quick
        )
        
        if crime_prediction_dataset is not None:
            # Save the temporal dataset
            output_file = f"crime_prediction_temporal_dataset_{args.window}h.pkl"
            crime_prediction_dataset.to_pickle(output_file)
            log.info(f"Temporal dataset saved to {output_file}")
            
            # Save a CSV sample for inspection
            sample_file = f"crime_dataset_sample_{args.window}h_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            sample_size = min(1000, len(crime_prediction_dataset))
            crime_prediction_dataset.sample(sample_size).to_csv(sample_file, index=False)
            log.info(f"Sample dataset saved to {sample_file}")
            
            # Display summary statistics
            log.info(f"\n=== DATASET SUMMARY ===")
            log.info(f"Total samples: {len(crime_prediction_dataset)}")
            log.info(f"Features: {crime_prediction_dataset.columns.tolist()}")
            
            target_col = f'target_{args.window}h'
            if target_col in crime_prediction_dataset.columns:
                target_summary = crime_prediction_dataset[target_col].value_counts()
                log.info(f"Target distribution: {dict(target_summary)}")
        else:
            log.error("Failed to create temporal dataset")
            
    except Exception as e:
        log.exception(f"Error during preprocessing: {str(e)}")
        raise
    
    log.info(f"Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
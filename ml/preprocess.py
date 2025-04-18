"""
Preprocessing script for crime prediction dataset
This script processes the crime data, police station locations, and grid cells to create a dataset suitable for machine learning models.
It includes the following steps:
1. Count crimes in each grid cell
2. Calculate distance to the nearest police station for each grid cell
3. Add time-based features (hour of the day, day of the week) 
4. Add crime type features (counts of each crime type in each cell)
5. Add demographic features (from barrios that intersect with each cell)
6. Create temporal training data with sliding windows
7. Generate target variables for different prediction windows
8. Save the final dataset to a GeoPackage file


REQUIRED LIBRARIES:
- pandas
- geopandas
- shapely
- sklearn
- numpy
- matplotlib
- seaborn
"""

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
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


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
    
    print(f"Downloading {filename} from GitHub...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Successfully downloaded {filename}")
            return True
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

# Data loading function for easier reuse across modules
def load_datasets(data_dir="../geodata"):
    """
    Load all datasets required for crime prediction.
    Downloads files from GitHub if they're not found locally.
    
    Returns:
        grid_gdf: H3 grid cells
        gdf_crimenes: Crime data
        gdf_police: Police station locations
        gdf_barrios: Barrios (neighborhoods) data
    """
    # Get the absolute path to the data directory based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, data_dir))
    
    print(f"Using data directory: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load grid data
    print("Loading grid data...")
    grid_path = os.path.join(data_dir, "hex_grid.gpkg")
    if not os.path.exists(grid_path):
        print(f"File not found at {grid_path}, downloading...")
        download_github_file("hex_grid.gpkg", grid_path)
    
    grid_gdf = gpd.read_file(grid_path)
    print(f"Loaded grid with {len(grid_gdf)} cells")
    
    # Load crime data
    print("Loading crime data...")
    crime_path = os.path.join(data_dir, "crime_data1.geojson")
    if not os.path.exists(crime_path):
        print(f"File not found at {crime_path}, downloading...")
        download_github_file("crime_data1.geojson", crime_path)
    
    gdf_crimenes = gpd.read_file(crime_path)
    print(f"Loaded crime data with {len(gdf_crimenes)} records")
    
    # Load police station data
    print("Loading police station data...")
    police_path = os.path.join(data_dir, "police1.geojson")
    if not os.path.exists(police_path):
        print(f"File not found at {police_path}, downloading...")
        download_github_file("police1.geojson", police_path)
    
    gdf_police = gpd.read_file(police_path)
    print(f"Loaded police data with {len(gdf_police)} stations")
    
    # Load barrios data
    print("Loading barrios data...")
    barrios_path = os.path.join(data_dir, "barrios_medellin1.geojson")
    if not os.path.exists(barrios_path):
        print(f"File not found at {barrios_path}, downloading...")
        download_github_file("barrios_medellin1.geojson", barrios_path)
    
    gdf_barrios = gpd.read_file(barrios_path)
    print(f"Loaded barrios data with {len(gdf_barrios)} neighborhoods")
    
    # Ensure all datasets have the same CRS (EPSG:4326 is standard for lat/lon)
    print("Checking and harmonizing coordinate reference systems (CRS)...")
    
    # Check if grid_gdf has a CRS, if not set it to EPSG:4326
    if grid_gdf.crs is None:
        print("Setting grid CRS to EPSG:4326")
        grid_gdf.crs = "EPSG:4326"
    else:
        # If it has a different CRS, transform to EPSG:4326
        if grid_gdf.crs != "EPSG:4326":
            print(f"Converting grid CRS from {grid_gdf.crs} to EPSG:4326")
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Convert crime data to GeoDataFrame if it's not already
    if not isinstance(gdf_crimenes, gpd.GeoDataFrame):
        print("Converting crime data to GeoDataFrame...")
        if 'geometry' not in gdf_crimenes.columns:
            # Look for coordinate columns
            if all(col in gdf_crimenes.columns for col in ['latitud', 'longitud']):
                print("Creating geometry from latitud/longitud columns")
                from shapely.geometry import Point
                geometry = [Point(lon, lat) for lon, lat in zip(gdf_crimenes['longitud'], gdf_crimenes['latitud'])]
                gdf_crimenes = gpd.GeoDataFrame(gdf_crimenes, geometry=geometry, crs="EPSG:4326")
            elif all(col in gdf_crimenes.columns for col in ['latitude', 'longitude']):
                print("Creating geometry from latitude/longitude columns")
                from shapely.geometry import Point
                geometry = [Point(lon, lat) for lon, lat in zip(gdf_crimenes['longitude'], gdf_crimenes['latitude'])]
                gdf_crimenes = gpd.GeoDataFrame(gdf_crimenes, geometry=geometry, crs="EPSG:4326")
            else:
                raise ValueError("Crime data must have coordinate columns (latitud/longitud or latitude/longitude)")
    
    # Make sure crime data is in EPSG:4326
    if gdf_crimenes.crs is None:
        print("Setting crime data CRS to EPSG:4326")
        gdf_crimenes.crs = "EPSG:4326"
    else:
        if gdf_crimenes.crs != "EPSG:4326":
            print(f"Converting crime data CRS from {gdf_crimenes.crs} to EPSG:4326")
            gdf_crimenes = gdf_crimenes.to_crs("EPSG:4326")
    
    # Make sure police station data is in EPSG:4326
    if gdf_police.crs is None:
        print("Setting police data CRS to EPSG:4326")
        gdf_police.crs = "EPSG:4326"
    else:
        if gdf_police.crs != "EPSG:4326":
            print(f"Converting police data CRS from {gdf_police.crs} to EPSG:4326")
            gdf_police = gdf_police.to_crs("EPSG:4326")
    
    # Make sure barrios data is in EPSG:4326
    if gdf_barrios.crs is None:
        print("Setting barrios data CRS to EPSG:4326")
        gdf_barrios.crs = "EPSG:4326"
    else:
        if gdf_barrios.crs != "EPSG:4326":
            print(f"Converting barrios data CRS from {gdf_barrios.crs} to EPSG:4326")
            gdf_barrios = gdf_barrios.to_crs("EPSG:4326")
    
    print("All datasets now have matching CRS: EPSG:4326")
    print("Datasets loaded successfully")
    return grid_gdf, gdf_crimenes, gdf_police, gdf_barrios

# 1. Contar crimenes en cada celda
def count_crimes_per_cell(grid_gdf, crime_gdf):
    """Count crimes in each grid cell"""
    joined = gpd.sjoin(grid_gdf, crime_gdf, how='left', predicate='intersects')
    crime_counts = joined.groupby('h3_index').size().reset_index(name='crime_count')

    grid_with_crimes = grid_gdf.merge(crime_counts, on='h3_index', how='left')
    grid_with_crimes['crime_count'] = grid_with_crimes['crime_count'].fillna(0)
    return grid_with_crimes

# 2. Calcular distancia a CAI mas cercano en cada celda
def add_distance_to_police(grid_gdf, police_gdf):
    """Calculate distance to nearest police station for each grid cell"""
    grid_centroids = grid_gdf.copy()
    
    if grid_centroids.crs != police_gdf.crs:
        print(f"Converting police stations CRS to match grid CRS: {grid_centroids.crs}")
        police_gdf = police_gdf.to_crs(grid_centroids.crs)
    
    #Medellín, Colombia (6°N, 75°W), UTM zone 18N
    print("Projecting to UTM Zone 18N (EPSG:32618) for accurate distance calculations")
    grid_projected = grid_centroids.to_crs("EPSG:32618")
    police_projected = police_gdf.to_crs("EPSG:32618")
    
    grid_projected['centroid'] = grid_projected.geometry.centroid
    
    # Calcular distancias en metros a CAIs
    print("Calculating distances to nearest police stations...")
    distances = []
    total_cells = len(grid_projected)
    
    for i, cell in grid_projected.iterrows():
        if i % 100 == 0:
            print(f"Processing cell {i}/{total_cells}")
            
        dist_to_police = police_projected.geometry.distance(cell['centroid'])
        min_dist = dist_to_police.min()
        distances.append(min_dist)
    
    grid_gdf['distance_to_police'] = distances
    
    # Convertir distancia a km
    grid_gdf['distance_to_police'] = grid_gdf['distance_to_police'] / 1000
    
    print(f"Distance calculations complete. Range: {grid_gdf['distance_to_police'].min():.2f} - {grid_gdf['distance_to_police'].max():.2f} km")
    return grid_gdf

# 3. Features temporales con ventanas deslizantes
def add_time_features(grid_gdf, crime_gdf, time_windows=None):
    """
    Add time-based crime features for each cell with multiple time windows
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        crime_gdf: GeoDataFrame with crime data including fecha_hecho (timestamp)
        time_windows: List of time windows in days to use for recent crime weighting
                      Default: [7, 30, 90] for 1 week, 1 month, 3 months
    
    Returns:
        GeoDataFrame with added time features
    """
    # Default time windows if not specified
    if time_windows is None:
        time_windows = [7, 30, 90]  # 1 week, 1 month, 3 months
    
    # Ensure both geodataframes have the same CRS
    if grid_gdf.crs != crime_gdf.crs:
        print(f"Ensuring CRS compatibility: {grid_gdf.crs} vs {crime_gdf.crs}")
        crime_gdf = crime_gdf.to_crs(grid_gdf.crs)
    
    # Create a copy of the grid_gdf for the results
    result = grid_gdf.copy()
    
    # Define time periods and days
    time_periods = {
        'morning': (6, 12),
        'afternoon': (12, 18),
        'evening': (18, 24),
        'night': (0, 6)
    }
    
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    for period in time_periods:
        result[f'crimes_{period}'] = 0
    
    for day in days:
        result[f'crimes_{day}'] = 0
    
    # Add time window features
    for window in time_windows:
        result[f'crimes_last_{window}d'] = 0
    
    crime_sindex = crime_gdf.sindex
    
    # Get the latest crime date to use as reference
    latest_date = crime_gdf['fecha_hecho'].max()
    print(f"Using latest crime date as reference: {latest_date}")
    
    # Process each cell
    total_cells = len(grid_gdf)
    for i, (idx, cell) in enumerate(grid_gdf.iterrows()):
        if i % 100 == 0:
            print(f"Processing time features for cell {i}/{total_cells}")
        
        # Use spatial index to find candidates
        possible_matches_index = list(crime_sindex.intersection(cell.geometry.bounds))
        if possible_matches_index:
            possible_matches = crime_gdf.iloc[possible_matches_index]
            # Confirm that the candidates actually intersect
            precise_matches = possible_matches[possible_matches.intersects(cell.geometry)]
            
            if len(precise_matches) > 0:
                # Add time period counts
                for period, (start_hour, end_hour) in time_periods.items():
                    period_crimes = precise_matches[
                        precise_matches['fecha_hecho'].dt.hour.between(start_hour, end_hour - 1)
                    ]
                    result.at[idx, f'crimes_{period}'] = len(period_crimes)
                
                # Add day of week counts
                for i, day in enumerate(days):
                    day_crimes = precise_matches[precise_matches['fecha_hecho'].dt.dayofweek == i]
                    result.at[idx, f'crimes_{day}'] = len(day_crimes)
                
                # Add time window counts with recency weighting
                for window in time_windows:
                    cutoff_date = latest_date - timedelta(days=window)
                    recent_crimes = precise_matches[precise_matches['fecha_hecho'] >= cutoff_date]
                    
                    # Apply recency weighting - more recent crimes have higher weight
                    if len(recent_crimes) > 0:
                        # Calculate days from cutoff for each crime
                        days_from_cutoff = (recent_crimes['fecha_hecho'] - cutoff_date).dt.total_seconds() / (24 * 3600)
                        # Apply exponential decay weight: more recent = higher weight
                        weights = np.exp(days_from_cutoff / window)  # Normalized by window size
                        weighted_count = weights.sum()
                        result.at[idx, f'crimes_last_{window}d'] = weighted_count
    
    print("Time features added successfully")
    return result

# 4. Features de tipo de crimen con ponderación temporal
def add_crime_type_features(grid_gdf, crime_gdf, time_windows=None):
    """
    Add crime type features for each cell with recency weighting
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        crime_gdf: GeoDataFrame with crime data
        time_windows: List of time windows in days for recency features
                      Default: [30, 90] for 1 month and 3 months
    """
    # Default time windows if not specified
    if time_windows is None:
        time_windows = [30, 90]  # 1 month, 3 months
        
    # Ensure both geodataframes have the same CRS
    if grid_gdf.crs != crime_gdf.crs:
        print(f"Ensuring CRS compatibility for crime types: {grid_gdf.crs} vs {crime_gdf.crs}")
        crime_gdf = crime_gdf.to_crs(grid_gdf.crs)
    
    # Get the latest crime date to use as reference
    latest_date = crime_gdf['fecha_hecho'].max()
    
    crime_types = crime_gdf['modalidad'].unique()
    print(f"Found {len(crime_types)} different crime types")
    
    # Create a copy of grid_gdf for the results
    result = grid_gdf.copy()
    
    # Initialize columns with zeros
    for crime_type in crime_types:
        # Standard count of each crime type
        column_name = f'crime_type_{crime_type}'
        result[column_name] = 0
        
        # Add time window columns for each crime type
        for window in time_windows:
            result[f'{column_name}_last_{window}d'] = 0
    
    # Create a spatial index for crime_gdf to improve performance
    crime_sindex = crime_gdf.sindex
    
    # Process each cell
    total_cells = len(grid_gdf)
    for i, (idx, cell) in enumerate(grid_gdf.iterrows()):
        if i % 100 == 0:
            print(f"Processing crime types for cell {i}/{total_cells}")
        
        # Use spatial index to find candidates
        possible_matches_index = list(crime_sindex.intersection(cell.geometry.bounds))
        if possible_matches_index:
            possible_matches = crime_gdf.iloc[possible_matches_index]
            # Confirm that the candidates actually intersect
            precise_matches = possible_matches[possible_matches.intersects(cell.geometry)]
            
            if len(precise_matches) > 0:
                # Count by crime type (overall)
                type_counts = precise_matches['modalidad'].value_counts()
                for crime_type, count in type_counts.items():
                    result.at[idx, f'crime_type_{crime_type}'] = count
                
                # Add time window counts for each crime type
                for window in time_windows:
                    cutoff_date = latest_date - timedelta(days=window)
                    recent_crimes = precise_matches[precise_matches['fecha_hecho'] >= cutoff_date]
                    
                    if len(recent_crimes) > 0:
                        # Calculate days from cutoff for each crime
                        recent_crimes['days_from_cutoff'] = (recent_crimes['fecha_hecho'] - cutoff_date).dt.total_seconds() / (24 * 3600)
                        
                        # Group by crime type and calculate weighted counts
                        for crime_type in crime_types:
                            type_crimes = recent_crimes[recent_crimes['modalidad'] == crime_type]
                            if len(type_crimes) > 0:
                                # Apply exponential decay weight
                                weights = np.exp(type_crimes['days_from_cutoff'] / window)
                                weighted_count = weights.sum()
                                result.at[idx, f'crime_type_{crime_type}_last_{window}d'] = weighted_count
    
    return result

# 5. features demograficas
def add_demographic_features(grid_gdf, barrios_gdf):
    """Add demographic features from barrios that intersect with each cell"""
    # barrios que intersectan en cada celda
    grid_with_barrios = grid_gdf.copy()
    grid_with_barrios['barrios'] = None

    for idx, cell in grid_gdf.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            grid_with_barrios.at[idx, 'barrios'] = ','.join(intersecting_barrios['nombre'].tolist())

    # comuna de cada celda
    grid_with_barrios['comuna'] = None

    for idx, cell in grid_gdf.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            # comuna mas comun que intersecta
            comuna = intersecting_barrios['codigo_comuna'].mode()[0]
            grid_with_barrios.at[idx, 'comuna'] = comuna

    return grid_with_barrios

# 6. Sliding window generator for temporal training
def generate_temporal_windows(grid_gdf, crime_gdf, window_sizes=None, step_size=24, quick_mode=False, max_windows=None, sample_size=100, recency_windows=None):
    """
    Generate temporal sliding windows for training
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        crime_gdf: GeoDataFrame with crime data
        window_sizes: List of future window sizes in hours for prediction targets
                     Default: [6, 12, 24, 72] for 6h, 12h, 24h, 3 days
        step_size: Hours to step forward for each new training window 
                   Default: 24 (daily windows)
        quick_mode: If True, process fewer windows and cells for faster execution
        max_windows: Maximum number of windows to process (for quick mode)
        sample_size: Number of grid cells to sample in quick mode (default: 100)
        recency_windows: List of past time windows in days for feature generation
                     Default: [1, 3, 7, 14, 30]
                   
    Returns:
        List of temporal window DataFrames, each with features and multiple target columns
    """
    if window_sizes is None:
        window_sizes = [6, 12, 24, 72]  # 6h, 12h, 24h, 3 days
    
    print(f"Generating temporal windows with sizes {window_sizes} hours and step size {step_size} hours")
    
    # Get date range from crime data
    start_date = crime_gdf['fecha_hecho'].min()
    end_date = crime_gdf['fecha_hecho'].max()
    
    # Calculate the minimum lookback period needed for features
    # In quick mode, we can use a shorter lookback period for faster processing
    if quick_mode:
        feature_lookback = timedelta(days=30)  # 30 days in quick mode 
        print(f"QUICK MODE: Using 30-day feature lookback instead of 90 days")
    else:
        feature_lookback = timedelta(days=90)  # 90 days in full mode
    
    # Adjust start date to allow for feature generation
    training_start = start_date + feature_lookback
    
    print(f"Data spans from {start_date} to {end_date}")
    print(f"Training windows will start from {training_start}")
    
    # Generate reference points (every step_size hours)
    reference_points = []
    current = training_start
    while current < end_date - timedelta(hours=max(window_sizes)):
        reference_points.append(current)
        current += timedelta(hours=step_size)
    
    # In quick mode, limit the number of windows
    if quick_mode:
        if max_windows is None:
            max_windows = 20  # Default if not specified
        print(f"QUICK MODE: Limiting to {max_windows} windows instead of {len(reference_points)}")
        # Take evenly spaced windows for better representation across the time range
        if len(reference_points) > max_windows:
            indices = np.linspace(0, len(reference_points) - 1, max_windows, dtype=int)
            reference_points = [reference_points[i] for i in indices]
    
    print(f"Generated {len(reference_points)} reference time points")
    
    # In quick mode, take a sample of the grid cells
    if quick_mode:
        actual_sample = min(sample_size, len(grid_gdf))
        print(f"QUICK MODE: Using {actual_sample} grid cells instead of {len(grid_gdf)}")
        grid_gdf = grid_gdf.sample(actual_sample, random_state=42)
    
    # Create spatial index for crime_gdf to improve performance
    print("Building spatial index for crime data...")
    # This explicitly creates the spatial index if it doesn't exist
    crime_gdf = crime_gdf.copy()
    crime_sindex = crime_gdf.sindex
    
    # Prepare the list of window DataFrames
    window_data = []
    
    # Define recency time windows from reference point if not provided
    if recency_windows is None:
        if quick_mode:
            recency_windows = [1, 7, 30]  # Simplified windows in quick mode
        else:
            recency_windows = [1, 3, 7, 14, 30]  # 1d, 3d, 7d, 14d, 30d
    
    # Process each reference point
    for i, ref_point in enumerate(reference_points):
        if i % 10 == 0 or i == len(reference_points) - 1:  # Progress update every 10 windows
            print(f"Processing window {i+1}/{len(reference_points)} at {ref_point}")
        
        # Create a copy of the grid for this time window
        window_gdf = grid_gdf.copy()
        
        # Convert timezone-aware datetime to naive datetime to avoid Excel export issues
        if ref_point.tzinfo is not None:
            naive_ref_point = ref_point.replace(tzinfo=None)
        else:
            naive_ref_point = ref_point
            
        window_gdf['reference_time'] = naive_ref_point
        
        # Filter crimes for feature generation (before reference point)
        past_crimes = crime_gdf[crime_gdf['fecha_hecho'] < ref_point]
        
        # Add the reference hour and day as features
        window_gdf['ref_hour'] = ref_point.hour
        window_gdf['ref_day'] = ref_point.dayofweek
        window_gdf['ref_month'] = ref_point.month
        window_gdf['ref_is_weekend'] = 1 if ref_point.dayofweek >= 5 else 0
        
        # Add recency-weighted crime counts
        for window in recency_windows:
            window_gdf[f'crimes_last_{window}d'] = 0
            cutoff = ref_point - timedelta(days=window)
            
            # Filter past crimes within this window
            window_crimes = past_crimes[past_crimes['fecha_hecho'] >= cutoff]
            if len(window_crimes) == 0:
                continue
                
            # Create a spatial index for the filtered crimes
            if hasattr(window_crimes, 'sindex') and window_crimes.sindex is not None:
                window_crimes_sindex = window_crimes.sindex
            else:
                window_crimes_sindex = window_crimes.sindex
            
            # Process each cell
            for idx, cell in window_gdf.iterrows():
                try:
                    # Use spatial index to find candidates, checking for errors
                    if window_crimes_sindex is not None:
                        possible_matches_index = list(window_crimes_sindex.intersection(cell.geometry.bounds))
                        
                        # Validate indices to avoid out-of-bounds errors
                        valid_indices = [i for i in possible_matches_index if i < len(window_crimes)]
                        if valid_indices:
                            possible_matches = window_crimes.iloc[valid_indices]
                            
                            # Confirm that the candidates actually intersect
                            recent_matches = possible_matches[possible_matches.intersects(cell.geometry)]
                            
                            if len(recent_matches) > 0:
                                # Calculate days from cutoff for each crime
                                days_from_cutoff = (recent_matches['fecha_hecho'] - cutoff).dt.total_seconds() / (24 * 3600)
                                # Apply exponential decay weight: more recent = higher weight
                                weights = np.exp(days_from_cutoff / window)  # Normalized by window size
                                weighted_count = weights.sum()
                                window_gdf.at[idx, f'crimes_last_{window}d'] = weighted_count
                except Exception as e:
                    if not quick_mode:  # Only print warnings in full mode
                        print(f"Warning: Error processing cell at index {idx} for window {window}d: {e}")
                    continue
        
        # Generate target variables for each prediction window
        for hours in window_sizes:
            future_cutoff = ref_point + timedelta(hours=hours)
            target_col = f'target_{hours}h'
            window_gdf[target_col] = 0
            
            # Filter future crimes for this window
            future_crimes = crime_gdf[
                (crime_gdf['fecha_hecho'] >= ref_point) & 
                (crime_gdf['fecha_hecho'] < future_cutoff)
            ]
            
            if len(future_crimes) == 0:
                continue
                
            # Create spatial index for future crimes
            if hasattr(future_crimes, 'sindex') and future_crimes.sindex is not None:
                future_crimes_sindex = future_crimes.sindex
            else:
                future_crimes_sindex = future_crimes.sindex
            
            # Process each cell
            for idx, cell in window_gdf.iterrows():
                try:
                    # Use spatial index to find candidates
                    if future_crimes_sindex is not None:
                        possible_matches_index = list(future_crimes_sindex.intersection(cell.geometry.bounds))
                        
                        # Validate indices to avoid out-of-bounds errors
                        valid_indices = [i for i in possible_matches_index if i < len(future_crimes)]
                        if valid_indices:
                            possible_matches = future_crimes.iloc[valid_indices]
                            
                            # Check if any future crimes intersect with this cell
                            has_crime = any(possible_matches.intersects(cell.geometry))
                            if has_crime:
                                window_gdf.at[idx, target_col] = 1
                except Exception as e:
                    if not quick_mode:  # Only print warnings in full mode
                        print(f"Warning: Error processing future crimes for cell at index {idx} for {hours}h window: {e}")
                    continue
        
        # Add this window to the collection
        window_data.append(window_gdf)
    
    print(f"Generated {len(window_data)} temporal windows")
    # If window_data is empty, return None
    if not window_data:
        return None
        
    # Combine all windows into a single DataFrame
    combined_windows = pd.concat(window_data, ignore_index=True)
    print(f"Combined temporal dataset shape: {combined_windows.shape}")
    
    # Print a sample of the data to verify
    print("\nSample of temporal dataset (first few rows, selected columns):")
    sample_cols = ['reference_time', 'ref_hour', 'ref_day'] + [f'crimes_last_{w}d' for w in recency_windows] + [f'target_{h}h' for h in window_sizes]
    sample_cols = [col for col in sample_cols if col in combined_windows.columns]
    print(combined_windows[sample_cols].head(3))
    
    return grid_gdf

# Create a comprehensive function to call all the preprocessing steps
def create_crime_prediction_dataset(grid_gdf, crime_gdf, police_gdf, barrios_gdf, temporal_windows=True, quick_mode=False):
    """
    Create a complete crime prediction dataset by applying all preprocessing steps
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        crime_gdf: GeoDataFrame with crime data
        police_gdf: GeoDataFrame with police station locations
        barrios_gdf: GeoDataFrame with neighborhood data
        temporal_windows: Whether to create temporal windows for prediction
        quick_mode: If True, process fewer data points for faster execution
    
    Returns:
        GeoDataFrame with all features for crime prediction
    """
    print("\n1. Counting crimes per grid cell...")
    grid_with_crimes = count_crimes_per_cell(grid_gdf, crime_gdf)
    print(f"Crime counts added. Range: {grid_with_crimes['crime_count'].min()} - {grid_with_crimes['crime_count'].max()}")
    
    print("\n2. Calculating distance to nearest police station...")
    grid_with_police = add_distance_to_police(grid_with_crimes, police_gdf)
    
    print("\n3. Adding time-based features...")
    # Use fewer time windows in quick mode
    time_windows = [7, 30] if quick_mode else [7, 30, 90]
    grid_with_time = add_time_features(grid_with_police, crime_gdf, time_windows=time_windows)
    
    print("\n4. Adding crime type features...")
    grid_with_crimes_types = add_crime_type_features(
        grid_with_time, 
        crime_gdf,
        time_windows=[30] if quick_mode else [30, 90]
    )
    
    print("\n5. Adding demographic features...")
    grid_with_demographics = add_demographic_features(grid_with_crimes_types, barrios_gdf)
    
    # Either return the static dataset or continue to create temporal windows
    if not temporal_windows:
        print("\nCreated static crime prediction dataset")
        return grid_with_demographics
    
    print("\n6. Generating temporal windows for prediction...")
    # Use fewer windows and a smaller sample in quick mode
    window_sizes = [24] if quick_mode else [6, 12, 24, 72]
    max_windows = 10 if quick_mode else None
    sample_size = 100 if quick_mode else 1000
    
    temporal_dataset = generate_temporal_windows(
        grid_with_demographics,
        crime_gdf,
        window_sizes=window_sizes,
        step_size=24,  # Daily windows
        quick_mode=quick_mode,
        max_windows=max_windows,
        sample_size=sample_size
    )
    
    print("\nTemporal crime prediction dataset created")
    return temporal_dataset

# Update the main function to include command line argument for quick mode
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess crime data for predictive modeling')
    parser.add_argument('--quick', action='store_true', help='Process a smaller subset of data for faster execution')
    args = parser.parse_args()
    
    print(f"Preprocessing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.quick:
        print("\n===== QUICK MODE ENABLED: Processing reduced dataset for rapid testing =====\n")
    
    # Load datasets
    grid_gdf, gdf_crimenes, gdf_police, gdf_barrios = load_datasets()
    
    # Create the complete dataset (with temporal windows)
    crime_prediction_dataset = create_crime_prediction_dataset(
        grid_gdf,
        gdf_crimenes,
        gdf_police,
        gdf_barrios,
        temporal_windows=True,
        quick_mode=args.quick
    )

    # Export the dataframe to Excel for analysis (sample only due to size)
    if crime_prediction_dataset is not None:
        # Convert timezone-aware datetime columns to timezone-naive for Excel compatibility
        crime_prediction_dataset = convert_datetime_columns_to_naive(crime_prediction_dataset)
        
        sample_size = min(1000, len(crime_prediction_dataset))
        crime_prediction_dataset.sample(sample_size).to_excel("crime_prediction_temporal_sample.xlsx", index=False)
        print(f"Sample of {sample_size} records exported to Excel for analysis")

        # Export the full dataset to a compressed format
        crime_prediction_dataset.to_pickle("crime_prediction_temporal_dataset.pkl")
        print("Full temporal dataset saved to crime_prediction_temporal_dataset.pkl")
    
    # Also create a static dataset for backward compatibility
    static_dataset = create_crime_prediction_dataset(
        grid_gdf,
        gdf_crimenes,
        gdf_police,
        gdf_barrios,
        temporal_windows=False,
        quick_mode=args.quick
    )
    
    # Export the spatial dataset to GeoPackage
    static_dataset.to_file("crime_prediction_grid.gpkg", driver="GPKG")
    print("Static dataset saved to crime_prediction_grid.gpkg")
    
    print(f"Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
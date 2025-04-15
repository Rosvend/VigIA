"""
Preprocessing script for crime prediction dataset
This script processes the crime data, police station locations, and grid cells to create a dataset suitable for machine learning models.
It includes the following steps:
1. Count crimes in each grid cell
2. Calculate distance to the nearest police station for each grid cell
3. Add time-based features (hour of the day, day of the week)
4. Add crime type features (counts of each crime type in each cell)
5. Add demographic features (from barrios that intersect with each cell)
6. Create a binary target variable indicating whether the crime count exceeds a certain threshold
7. Save the final dataset to a GeoPackage file


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
from datetime import datetime

# Data loading function for easier reuse across modules
def load_datasets(data_dir="../geodata"):
    """
    Load all datasets required for crime prediction.
    
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
    
    print("Loading grid data...")
    grid_path = os.path.join(data_dir, "hex_grid.gpkg")
    if not os.path.exists(grid_path):
        print(f"Warning: File not found at {grid_path}")
        # Try alternative paths
        alt_path = os.path.join(os.path.dirname(script_dir), "geodata", "hex_grid.gpkg")
        if os.path.exists(alt_path):
            grid_path = alt_path
            print(f"Using alternative path: {grid_path}")
    
    grid_gdf = gpd.read_file(grid_path)
    
    print("Loading crime data...")
    crime_path = os.path.join(data_dir, "crime_data1.geojson")
    if not os.path.exists(crime_path):
        print(f"Warning: File not found at {crime_path}")
        alt_path = os.path.join(os.path.dirname(script_dir), "geodata", "crime_data1.geojson")
        if os.path.exists(alt_path):
            crime_path = alt_path
            print(f"Using alternative path: {crime_path}")
    
    gdf_crimenes = gpd.read_file(crime_path)
    
    print("Loading police station data...")
    police_path = os.path.join(data_dir, "police1.geojson")
    if not os.path.exists(police_path):
        print(f"Warning: File not found at {police_path}")
        alt_path = os.path.join(os.path.dirname(script_dir), "geodata", "police1.geojson")
        if os.path.exists(alt_path):
            police_path = alt_path
            print(f"Using alternative path: {police_path}")
    
    gdf_police = gpd.read_file(police_path)
    
    print("Loading barrios data...")
    barrios_path = os.path.join(data_dir, "barrios_medellin1.geojson")
    if not os.path.exists(barrios_path):
        print(f"Warning: File not found at {barrios_path}")
        alt_path = os.path.join(os.path.dirname(script_dir), "geodata", "barrios_medellin1.geojson")
        if os.path.exists(alt_path):
            barrios_path = alt_path
            print(f"Using alternative path: {barrios_path}")
    
    gdf_barrios = gpd.read_file(barrios_path)
    
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

# 3. features de tiempo
def add_time_features(grid_gdf, crime_gdf):
    """Add time-based crime features for each cell"""
    if grid_gdf.crs != crime_gdf.crs:
        print(f"Ensuring CRS compatibility: {grid_gdf.crs} vs {crime_gdf.crs}")
        crime_gdf = crime_gdf.to_crs(grid_gdf.crs)
    
    result = grid_gdf.copy()
    
    # periodos de tiempo
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
    
    crime_sindex = crime_gdf.sindex
    
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
    
    return result

# 4. features de crimen
def add_crime_type_features(grid_gdf, crime_gdf):
    """Add crime type features for each cell"""
    if grid_gdf.crs != crime_gdf.crs:
        print(f"Ensuring CRS compatibility for crime types: {grid_gdf.crs} vs {crime_gdf.crs}")
        crime_gdf = crime_gdf.to_crs(grid_gdf.crs)
    
    crime_types = crime_gdf['modalidad'].unique()
    print(f"Found {len(crime_types)} different crime types")
    
    result = grid_gdf.copy()
    
    for crime_type in crime_types:
        column_name = f'crime_type_{crime_type}'
        result[column_name] = 0
    
    crime_sindex = crime_gdf.sindex
    
    total_cells = len(grid_gdf)
    for i, (idx, cell) in enumerate(grid_gdf.iterrows()):
        if i % 100 == 0:
            print(f"Processing crime types for cell {i}/{total_cells}")
        
        possible_matches_index = list(crime_sindex.intersection(cell.geometry.bounds))
        if possible_matches_index:
            possible_matches = crime_gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(cell.geometry)]
            
            if len(precise_matches) > 0:
                type_counts = precise_matches['modalidad'].value_counts()
                for crime_type, count in type_counts.items():
                    result.at[idx, f'crime_type_{crime_type}'] = count
    
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

# 6. Crear target variable
def create_target_variable(grid_gdf, crime_gdf, threshold=0):
    """Create binary target variable - 1 if crime count >= threshold, 0 otherwise"""
    grid_gdf['target'] = grid_gdf['crime_count'] > threshold
    grid_gdf['target'] = grid_gdf['target'].astype(int)
    return grid_gdf

def create_crime_prediction_dataset(grid_gdf, crime_gdf, police_gdf, barrios_gdf):
    """
    Create complete dataset for crime prediction with all preprocessing steps.
    Standardized function for use in both training and prediction modules.
    """
    print("Creating crime prediction dataset...")
    print("1. Counting crimes per cell...")
    grid_gdf = count_crimes_per_cell(grid_gdf, crime_gdf)
    
    print("2. Adding distance to police...")
    grid_gdf = add_distance_to_police(grid_gdf, police_gdf)
    
    print("3. Adding time features...")
    grid_gdf = add_time_features(grid_gdf, crime_gdf)
    
    print("4. Adding crime type features...")
    grid_gdf = add_crime_type_features(grid_gdf, crime_gdf)
    
    print("5. Adding demographic features...")
    grid_gdf = add_demographic_features(grid_gdf, barrios_gdf)
    
    print("6. Creating target variable...")
    grid_gdf = create_target_variable(grid_gdf, crime_gdf, threshold=5)
    
    # Print a sample of the dataset to verify
    print("\nSample of preprocessed dataset (first few columns):")
    sample_cols = ['h3_index', 'crime_count', 'distance_to_police', 'target'] 
    print(grid_gdf[sample_cols].head(3))
    
    return grid_gdf


if __name__ == "__main__":
    print(f"Preprocessing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    grid_gdf, gdf_crimenes, gdf_police, gdf_barrios = load_datasets()
    
    crime_prediction_dataset = create_crime_prediction_dataset(
        grid_gdf,
        gdf_crimenes,
        gdf_police,
        gdf_barrios
    )

    crime_prediction_dataset.to_excel("crime_prediction_dataset.xlsx", index=False)
    print("Dataset exported to Excel for analysis")

    X = crime_prediction_dataset.drop(['geometry', 'target'], axis=1)
    y = crime_prediction_dataset['target']

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")

    # Export the spatial dataset to GeoPackage
    crime_prediction_dataset.to_file("crime_prediction_grid.gpkg", driver="GPKG")
    print("Dataset saved to crime_prediction_grid.gpkg")
    
    print(f"Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
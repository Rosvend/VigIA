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

# 1. Contar crimenes en cada celda
def count_crimes_per_cell(grid_gdf, crime_gdf):
    """Count crimes in each grid cell"""
    # Spatial join to count crimes in each cell
    joined = gpd.sjoin(grid_gdf, crime_gdf, how='left', predicate='intersects')
    crime_counts = joined.groupby('cell_id').size().reset_index(name='crime_count')

    # Merge back with grid_gdf
    grid_with_crimes = grid_gdf.merge(crime_counts, on='cell_id', how='left')
    grid_with_crimes['crime_count'] = grid_with_crimes['crime_count'].fillna(0)
    return grid_with_crimes

# 2. Calcular distancia a CAI mas cercano en cada celda
def add_distance_to_police(grid_gdf, police_gdf):
    """Calculate distance to nearest police station for each grid cell"""
    # centroide de cada celda
    grid_centroids = grid_gdf.copy()
    grid_centroids['centroid'] = grid_centroids.geometry.centroid

    # distancias a CAI
    distances = []
    for _, cell in grid_centroids.iterrows():
        dist_to_police = police_gdf.geometry.distance(cell['centroid'])
        min_dist = dist_to_police.min()
        distances.append(min_dist)

    grid_gdf['distance_to_police'] = distances
    return grid_gdf

# 3. features de tiempo
def add_time_features(grid_gdf, crime_gdf):
    """Add time-based crime features for each cell"""
    grid_df = pd.DataFrame(grid_gdf.drop('geometry', axis=1))

    time_periods = {
        'morning': (6, 12),
        'afternoon': (12, 18),
        'evening': (18, 24),
        'night': (0, 6)
    }

    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    for period in time_periods:
        grid_df[f'crimes_{period}'] = 0

    for day in days:
        grid_df[f'crimes_{day}'] = 0

    for _, cell in grid_gdf.iterrows():
        cell_id = cell['cell_id']
        crimes_in_cell = gpd.sjoin(gpd.GeoDataFrame([cell]), crime_gdf, how='left', predicate='intersects')

        if len(crimes_in_cell) > 0:
            for period, (start_hour, end_hour) in time_periods.items():
                period_crimes = crimes_in_cell[crimes_in_cell['fecha_hecho'].dt.hour.between(start_hour, end_hour)]
                grid_df.loc[grid_df['cell_id'] == cell_id, f'crimes_{period}'] = len(period_crimes)

            for i, day in enumerate(days):
                day_crimes = crimes_in_cell[crimes_in_cell['fecha_hecho'].dt.dayofweek == i]
                grid_df.loc[grid_df['cell_id'] == cell_id, f'crimes_{day}'] = len(day_crimes)

    result = grid_gdf.merge(grid_df.drop('cell_id', axis=1), left_index=True, right_index=True)
    return result

# 4. features de crimen
def add_crime_type_features(grid_gdf, crime_gdf):
    """Add crime type features for each cell"""
    crime_types = crime_gdf['modalidad'].unique()

    grid_df = pd.DataFrame(grid_gdf.drop('geometry', axis=1))
    for crime_type in crime_types:
        grid_df[f'crime_type_{crime_type}'] = 0

    # Contar tipo de crimen en cada celda
    for _, cell in grid_gdf.iterrows():
        cell_id = cell['cell_id']
        crimes_in_cell = gpd.sjoin(gpd.GeoDataFrame([cell]), crime_gdf, how='left', predicate='intersects')

        if len(crimes_in_cell) > 0:
            # Contar por tipo de crimen
            type_counts = crimes_in_cell['modalidad'].value_counts()
            for crime_type, count in type_counts.items():
                grid_df.loc[grid_df['cell_id'] == cell_id, f'crime_type_{crime_type}'] = count

    result = grid_gdf.merge(grid_df.drop('cell_id', axis=1), left_index=True, right_index=True)
    return result

# 5. Add demographic features
def add_demographic_features(grid_gdf, barrios_gdf):
    """Add demographic features from barrios that intersect with each cell"""
    # Get barrios that intersect with each cell
    grid_with_barrios = grid_gdf.copy()
    grid_with_barrios['barrios'] = None

    for idx, cell in grid_gdf.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            grid_with_barrios.at[idx, 'barrios'] = ','.join(intersecting_barrios['nombre'].tolist())

    # Get comuna for each cell (most common comuna that intersects)
    grid_with_barrios['comuna'] = None

    for idx, cell in grid_gdf.iterrows():
        intersecting_barrios = barrios_gdf[barrios_gdf.geometry.intersects(cell.geometry)]
        if len(intersecting_barrios) > 0:
            # Get most common comuna
            comuna = intersecting_barrios['codigo_comuna'].mode()[0]
            grid_with_barrios.at[idx, 'comuna'] = comuna

    return grid_with_barrios

# 6. Crear target variable
def create_target_variable(grid_gdf, crime_gdf, threshold=0):
    """Create binary target variable - 1 if crime count >= threshold, 0 otherwise"""
    grid_gdf['target'] = grid_gdf['crime_count'] > threshold
    grid_gdf['target'] = grid_gdf['target'].astype(int)
    return grid_gdf

# Main function to create the complete dataset
def create_crime_prediction_dataset(grid_gdf, crime_gdf, police_gdf, barrios_gdf):
    """Create complete dataset for crime prediction"""
    # 1. Count crimes in each cell
    grid_gdf = count_crimes_per_cell(grid_gdf, crime_gdf)

    # 2. Calculate distance to nearest police station
    grid_gdf = add_distance_to_police(grid_gdf, police_gdf)

    # 3. Add time-based features
    grid_gdf = add_time_features(grid_gdf, crime_gdf)

    # 4. Add crime type features
    grid_gdf = add_crime_type_features(grid_gdf, crime_gdf)

    # 5. Add demographic features
    grid_gdf = add_demographic_features(grid_gdf, barrios_gdf)

    # 6. Create target variable (customize threshold based on your needs)
    grid_gdf = create_target_variable(grid_gdf, crime_gdf, threshold=5)

    return grid_gdf


crime_prediction_dataset = create_crime_prediction_dataset(
    grid_gdf,
    gdf_crimenes,
    gdf_police,
    gdf_barrios
)


X = crime_prediction_dataset.drop(['geometry', 'target'], axis=1)
y = crime_prediction_dataset['target']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Target distribution: {y.value_counts(normalize=True)}")

crime_prediction_dataset.to_file("crime_prediction_grid.gpkg", driver="GPKG")
print("Dataset saved to crime_prediction_grid.gpkg")
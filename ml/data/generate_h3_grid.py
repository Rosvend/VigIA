'''
PYTHON SCRIPT FOR GENERATING H3 GRID

This script generates a grid of hexagons using the H3 library (version 4.x).
It creates a GeoDataFrame with the hexagons and their centroids.
The hexagons are stored in a GeoDataFrame and can be used for
further analysis or visualization.
'''

import h3
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import argparse
import os
import requests

def download_geojson_if_needed(url, local_path='barrios.geojson'):
    """Download the GeoJSON file if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"Downloading {local_path} from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded to {local_path}")
            return local_path
        else:
            print(f"Failed to download: HTTP {response.status_code}")
            return None
    else:
        print(f"Using existing local file: {local_path}")
        return local_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate H3 hexagon grid from barrios data')
    parser.add_argument('--input', default='barrios.geojson', help='Path to input GeoJSON file with barrios data')
    parser.add_argument('--github_url', 
                   default='https://raw.githubusercontent.com/Rosvend/Patrol-routes-optimization-Medellin/main/geodata/barrios_medellin.geojson',
                   help='URL to download the GeoJSON if not available locally')
    parser.add_argument('--output', default='hex_grid.gpkg', help='Path to output GPKG file')
    parser.add_argument('--resolution', type=int, default=9, help='H3 resolution (0-15)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    
    args = parser.parse_args()
    
    # Download the file if needed
    input_file = args.input
    if not os.path.exists(input_file):
        input_file = download_geojson_if_needed(args.github_url, args.input)
        if input_file is None:
            raise FileNotFoundError(f"Could not find or download {args.input}")
    
    # Load input data
    print(f"Loading barrios from {input_file}")
    gdf_barrios = gpd.read_file(input_file)
    gdf_barrios = gdf_barrios.to_crs(epsg=4326)
    
    # Create city boundary
    print(f"Creating city boundary from {len(gdf_barrios)} neighborhoods")
    city_boundary = gdf_barrios.geometry.union_all()
    
    if city_boundary.is_empty:
        raise ValueError("City boundary is empty after union!")
    
    # Convert to H3 format
    geo_dict = city_boundary.__geo_interface__
    
    print(f"Generating H3 cells at resolution {args.resolution}")
    hexagons = h3.geo_to_cells(geo_dict, args.resolution)
    
    if not hexagons:
        raise ValueError("No hexagons generated - check input geometry and coordinates")
    
    print(f"Generated {len(hexagons)} hexagons")
    
    # Create geometries for each hexagon
    hex_geoms = [
        Polygon(
            [(lng, lat) for lat, lng in h3.cell_to_boundary(h)]
        ) for h in hexagons
    ]
    
    # Create GeoDataFrame
    hex_grid = gpd.GeoDataFrame(
        {"h3_index": hexagons, "geometry": hex_geoms},
        crs="EPSG:4326"
    ).dropna()
    
    # Save output
    print(f"Saving hex grid to {args.output}")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    hex_grid.to_file(args.output, driver="GPKG")
    
    # Generate visualization if requested
    if args.visualize:
        print("Generating visualization")
        fig, ax = plt.subplots(figsize=(12, 12))
        hex_grid.plot(
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_aspect('equal')
        ax.set_title(f"H3 Grid (Resolution {args.resolution})")
        
        # Save visualization to same directory as output file
        viz_path = os.path.splitext(args.output)[0] + ".png"
        plt.savefig(viz_path, dpi=300)
        plt.close()
        print(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    main()
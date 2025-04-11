from shapely import Point, Polygon, intersection
import numpy as np
from itertools import chain
import geopandas as gpd
from random import uniform

# TODO: find the ideal value for this variable
SCALE_FACTOR = 0.002
MAX_DISTANCE = 0.05

grid = gpd.GeoDataFrame.from_file("../geodata/grid.geojson").to_crs(crs="EPSG:4326")

class Hotarea:
    area: Polygon
    probability: float

    def __init__(self, probability, area: Polygon):
        self.area = area
        self.probability = probability

    def meshgrid(self):
        """
        Get a grid with Points across the spanning area with density
        proportional to probability.
        """
        spacing = SCALE_FACTOR / self.probability
        return [(x, y)
            for x in np.arange(self.area.bounds[0], self.area.bounds[2], spacing)
            for y in np.arange(self.area.bounds[1], self.area.bounds[3], spacing)
        ]

    def hotspots(self):
        return [ Hotspot(self.probability, x, y)
            for (x,y) in self.meshgrid()
            if Point(x,y).within(self.area)
        ]

class Hotspot:
    pt: Point
    probability: float
    
    def __init__(self, probability, *args):
        self.pt = Point(*args)
        self.probability = probability

    def toDict(self):
        return {
            'coordinates': [self.pt.x, self.pt.y],
            'probability': self.probability
        }

def stub_assign_probabilities(grid: gpd.GeoDataFrame) -> list:
    """A stub method to assign random probabilities to the set of
    geometries on a grid. Should be replaced by query to the model."""

    grid['probability'] = [uniform(0,1) for _ in grid.itertuples()]

def grid_intersection(op_area: Polygon) -> gpd.GeoSeries:
    return grid[grid.geometry.intersects(op_area)].copy()

def stub_random_hotspots_and_areas(op_area: Polygon, p_threshold: float = 0.0) -> list:
    hotspots = []
    areas = grid_intersection(op_area)
    stub_assign_probabilities(areas)
    for subarea in areas.itertuples():
        if subarea.probability >= p_threshold:
            hotspots.extend(
                Hotarea(subarea.probability, subarea.geometry)
                    .hotspots()
            )
    return hotspots, areas

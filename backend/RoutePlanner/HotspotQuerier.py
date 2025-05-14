from .prediction.wrapper_interface import ModelWrapperInterface
from itertools import chain
from random import uniform
from shapely import Point, Polygon, intersection, centroid, MultiPoint
import geopandas as gpd
import numpy as np
import itertools

# TODO: find the ideal value for this variable
SCALE_FACTOR = 0.0002
MAX_DISTANCE = 0.05
LUMPING_FACTOR = 20

grid = gpd.GeoDataFrame.from_file("../geodata/hex_grid.gpkg").to_crs(crs="EPSG:4326")

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
    
    def _inner_distance(self):
        return abs(self.area.bounds[2] - self.area.bounds[0]) / 4

    def _hotspot_number(self):
        return 1 + round(self.probability * LUMPING_FACTOR)

    def hotspots(self):
        center = centroid(self.area)
        n = self._hotspot_number()
        if n == 1:
            return [Hotspot(self.probability, center)]
        d = self._inner_distance()
        return [
            Hotspot(self.probability, center.x + d*np.cos(ang), center.y + d*np.sin(ang))
                for ang in np.linspace(0, np.pi*2, n, endpoint=False)
        ]

class Hotspot:
    pt: Point
    requested: bool
    probability: float
    
    def __init__(self, probability: float | None, *args):
        self.pt = Point(*args)
        if probability is None:
            self.probability = 1
            self.requested = True
        else:
            self.probability = probability
            self.requested = False

    def toDict(self):
        return {
            'coordinates': [self.pt.x, self.pt.y],
            'requested' : self.requested,
            'probability': self.probability
        }

def stub_assign_probabilities(grid: gpd.GeoDataFrame) -> list:
    """A stub method to assign random probabilities to the set of
    geometries on a grid. Should be replaced by query to the model."""

    grid['probability'] = [uniform(0,1) for _ in grid.itertuples()]

def grid_intersection(op_area: Polygon) -> gpd.GeoSeries:
    return grid[grid.geometry.intersects(op_area)].copy()

def gen_hotspots_and_areas(model_wrapper: ModelWrapperInterface, op_area: Polygon, p_threshold: float = 0.0, point_limit: int = None) -> list:
    hotspots = []
    areas = grid_intersection(op_area)
    areas = model_wrapper.predict(areas)
    areas = areas.sort_values("probability", ascending=False)
    for subarea in areas.itertuples():
        if point_limit and len(hotspots) > point_limit:
            break
        if subarea.probability >= p_threshold:
            cur_hotspots = Hotarea(subarea.probability, subarea.geometry).hotspots()
            hotspots.extend(cur_hotspots)
    
    return hotspots[:point_limit] if point_limit else hotspots, areas

from .HotspotQuerier import gen_hotspots_and_areas, Hotspot
from .ors_secrets import ORS_KEY
from .prediction.wrapper_interface import ModelWrapperInterface
from geopandas import GeoDataFrame
from shapely import Polygon, Point
import geopandas as gpd
import math
import openrouteservice
from openrouteservice.optimization import Job, Vehicle

AREA = 'Medellín, Colombia'
POLICE_STATIONS_FILE = '../geodata/police.geojson'
COMUNAS_FILE = '../geodata/comunas.geojson'
DEFAULT_ORS_PROFILE = 'driving-car'
ORS_PROFILES = ["driving-car", "driving-hgv", "foot-walking",
                "foot-hiking", "cycling-regular",
                "cycling-road","cycling-mountain",
                "cycling-electric",]
MAX_POINTS_PER_ROUTE = 50

def center(points: list):
    return(Point(
        sum([point.x for point in points]) / len(points),
        sum([point.y for point in points]) / len(points)
    ))

def angle(source: Point, dest: Point) -> float:
    return math.atan2(dest.y - source.y, dest.x - source.x) % (2*math.pi)


def classify_points(hotspots: list, n: int, start = None):
    """
    Classify the LonLat points of hotspots in n groups based on their
    angles from start, or the euclidian center, if not supplied.
    """
    if start is None:
        start = center([hotspot.pt for hotspot in hotspots])

    min_angle = min(angle(start, point.pt) for point in hotspots)
    max_angle = max(angle(start, point.pt) for point in hotspots)
    range_angles = max_angle - min_angle

    return [[
        point for point in hotspots
            if ((angle(start, point.pt) - min_angle) // (range_angles/n) == i)
        ] for i in range(n)]

# Stub?
comunas = gpd.read_file(COMUNAS_FILE)
def get_operation_area(police_station: Point) -> Polygon:
    return comunas.loc[comunas.contains(police_station)].union_all()

def pt2tup(point):
    if type(point) is Hotspot:
        point = point.pt
    return (point.x, point.y)

def filter_most_likely(points: list, include_station: bool) -> list:
    """Filter the the incoming list of points to make sure it has less
    points than the maximum permitted by ORS."""

    n_points = MAX_POINTS_PER_ROUTE - (2 * int(include_station))

    return sorted(points,
        key=(lambda pt: pt.probability),
        reverse=True
        )[:n_points]

class PoliceRouter:
    _ors_key: str
    _stations: GeoDataFrame
    _route_client: openrouteservice.Client
    _model_wrapper: ModelWrapperInterface

    def __init__(self, ors_key, model_wrapper):
        self._model_wrapper = model_wrapper
        self._stations = gpd.read_file(POLICE_STATIONS_FILE)
        self._route_client = openrouteservice.Client(key=ors_key)
        self._ors_key = ors_key
    
    def query_route(self, station, points, profile, include_station):
        result = self._route_client.directions(
                [pt2tup(point) for point in
                    ([station, *filter_most_likely(points, True), station]
                        if include_station
                        else filter_most_likely(points, False))
                ],
                profile=profile,
                radiuses=-1,
                optimize_waypoints=True
            )
        return openrouteservice.convert.decode_polyline(
            results['routes'][0]['geometry']
        )['coordinates']

    def query_routes(self, n, station, points, profile, include_station):
        result = self._route_client.optimization(
            jobs=[Job(id=idx,
                      location=pt2tup(point),
                      amount=[1])

                  for idx, point in enumerate(points)],
            vehicles=[Vehicle(id=i,
                              start=(pt2tup(station) if include_station else None),
                              end=(pt2tup(station) if include_station else None),
                              capacity=[round(len(points) / n)])
                      for i in range(n)],
            geometry=True)
        return [openrouteservice.convert.decode_polyline(route['geometry'])['coordinates']
                for route in result['routes']]

    def compute_routes(self,
                      cai_id: int,
                      n: int,
                      profile = DEFAULT_ORS_PROFILE,
                      include_station: bool = True,
                      threshold: float = 0.0,
                      include_hotspots: bool = False,
                      requested_spots: list[list] = []):
        
        station = self._stations.iloc[[cai_id]].geometry.union_all()
        area = get_operation_area(station)
        hotspots, areas = gen_hotspots_and_areas(self._model_wrapper, area, threshold, MAX_POINTS_PER_ROUTE - len(requested_spots))
        hotspots = [Hotspot(None, *pt) for pt in requested_spots] + hotspots
        # TODO: ensure the number of hotspots is non-zero
        # hotspot_areas = classify_points(hotspots, n, station)
        result = {
            'hotareas': areas.to_geo_dict(),
            'routes': self.query_routes(n, station, hotspots,
                                        profile, include_station)
                
        }
        if include_hotspots:
            result['hotspots'] = [hotspot.toDict() for hotspot in hotspots]
        return result

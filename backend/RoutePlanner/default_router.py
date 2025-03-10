import math
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Polygon, Point
from ors_secrets import ORS_KEY
from HotspotQuerier import stub_random_hotspots
import openrouteservice

AREA = 'Medellín, Colombia'
POLICE_STATIONS_FILE = '../geodata/police.geojson'
COMUNAS_FILE = '../geodata/comunas.geojson'
DEFAULT_ORS_PROFILE = 'driving-car'

def center(points: list):
    return(Point(
        sum([point.x for point in points]) / len(points),
        sum([point.y for point in points]) / len(points)
    ))

def angle(source: Point, dest: Point) -> float:
    return math.atan2(dest.y - source.y, dest.x - source.x) % (2*math.pi)


def classify_points(hotspots: list, n: int, start: LonLat = None):
    """
    Classify the LonLat points of hotspots in n groups based on their
    angles from start, or the euclidian center, if not supplied.
    """
    if start is None:
        start = center(hotspots)

    return [[
        point for point in hotspots
            if (angle(start, point) // (2*math.pi/n) == i)
        ] for i in range(n)]

# Stub?
comunas = gpd.read_file(COMUNAS_FILE)
def get_operation_area(police_station: Point) -> Polygon:
    return comunas.loc[comunas.contains(police_station)].union_all()

def pt2tup(point: Point):
    return (point.x, point.y)

class PoliceRouter:
    _stations: GeoDataFrame
    _route_client: openrouteservice.Client

    def __init__(self):
        self._stations = gpd.read_file(POLICE_STATIONS_FILE)
        self._route_client = openrouteservice.Client(key=ORS_KEY)
    
    def query_route(self, station, points):
        return openrouteservice.convert.decode_polyline(
            self._route_client.directions(
                [pt2tup(point) for point in [station, *points, station]],
                profile=DEFAULT_ORS_PROFILE,
                optimize_waypoints=True
            )['routes'][0]['geometry']
        )

    def compute_routes(self, cai_id: int, n: int):
        station = self._stations.iloc[[cai_id]].geometry.union_all()
        area = get_operation_area(station)
        hotspots = stub_random_hotspots(area)
        hotspot_areas = classify_points(hotspots, n)
        return [self.query_route(station, area)
            for area in hotspot_areas]

if __name__ == "__main__":
    # Test stuff, get rid of this please
    p_router = PoliceRouter()
    print(p_router.compute_routes(0, 2))

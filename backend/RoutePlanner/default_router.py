import math

class LonLat:
    lon: float
    lat: float
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat
    
    def angle(self, other):
        return math.atan2(other.lat - self.lat,other.lon - self.lon) % (2*math.pi)

    def center(points: list):
        return(LonLat(
            sum([point.lon for point in points]) / len(points),
            sum([point.lat for point in points]) / len(points)
        ))

def classify_points(hotspots: list, n: int, start: LonLat = None):
    """
    Classify the LonLat points of hotspots in n groups based on their
    angles from start, or the euclidian center, if not supplied.
    """
    if start is None:
        start = LonLat.center(hotspots)

    return [[
        point for point in hotspots
            if (start.angle(point) // (2*math.pi/n) == i)
        ] for i in range(n)]

class PoliceRouter:
    def compute_routes(self):
        # TODO: implement route computation
        return [[(-1.0, 4.0), (3.0, 0.5), (2.5, 9.0)]]

    

from shapely import Point, Polygon
from random import uniform

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

# A stup hotspot generator that generates hotspots randomly (a la montecarlo). Will be
# replaced with the code that actually queries the model.
def stub_random_hotspots(op_area: Polygon, n: int = 10) -> list:
    hotspots = []
    while len(hotspots) < n:
        point = Hotspot(
            uniform(0,1),
            uniform(op_area.bounds[0], op_area.bounds[2]),
            uniform(op_area.bounds[1], op_area.bounds[3])
        )
        if op_area.contains(point.pt):
            hotspots.append(point)
    return hotspots

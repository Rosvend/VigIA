from shapely import Point, Polygon, intersection
import numpy as np
from itertools import chain
from random import uniform

SCALE_FACTOR = 0.01
MAX_DISTANCE = 0.05

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


def random_point(area: Polygon) -> Point:
    point = None
    while point is None or not point.within(area):
        point = Point(uniform(area.bounds[0], area.bounds[2]),
                      uniform(area.bounds[1], area.bounds[3]))
    return point

def random_subarea(area: Polygon) -> Polygon:
    center = random_point(area)
    # create a random 4-vertex polygon from center
    return intersection(Polygon((
        (center.x + uniform(0, MAX_DISTANCE), center.y + uniform(0, MAX_DISTANCE)),
        (center.x + uniform(0, MAX_DISTANCE), center.y - uniform(0, MAX_DISTANCE)),
        (center.x - uniform(0, MAX_DISTANCE), center.y - uniform(0, MAX_DISTANCE)),
        (center.x - uniform(0, MAX_DISTANCE), center.y + uniform(0, MAX_DISTANCE)),
       )), area)

# A stub hotarea generator that generates a list of hotareas which are
# included within the given op_area. Will be replaced by code that
# actually queries the model
def stub_random_hotareas(op_area: Polygon, n: int = 10) -> list:
    return [
        Hotarea(uniform(0,1), random_subarea(op_area)) for _ in range(n)
    ]

def stub_random_hotspots(op_area: Polygon) -> list:
    return list(chain.from_iterable([subarea.hotspots() for subarea in stub_random_hotareas(op_area)]))

    

# A stup hotspot generator that generates hotspots randomly (a la montecarlo). Will be
# replaced with the code that actually queries the model.
#def stub_random_hotspots(op_area: Polygon, n: int = 10) -> list:
#    hotspots = []
#    while len(hotspots) < n:
#        point = Hotspot(
#            uniform(0,1),
#            uniform(op_area.bounds[0], op_area.bounds[2]),
#            uniform(op_area.bounds[1], op_area.bounds[3])
#        )
#        if op_area.contains(point.pt):
#            hotspots.append(point)
#    return hotspots

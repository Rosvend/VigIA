from shapely import Point, Polygon
from random import uniform

# A stup hotspot generator that generates hotspots randomly (a la montecarlo). Will be
# replaced with the code that actually queries the model.
def stub_random_hotspots(op_area: Polygon, n: int = 10) -> list:
    hotspots = []
    print(op_area.bounds[0])
    while len(hotspots) < n:
        point = Point(
            uniform(op_area.bounds[0], op_area.bounds[2]),
            uniform(op_area.bounds[1], op_area.bounds[3])
        )
        if op_area.contains(point):
            hotspots.append(point)
    return hotspots

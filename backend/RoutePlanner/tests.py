from . import default_router
from . import HotspotQuerier as HQ
from shapely import Polygon, Point, transform
import unittest

class TestClassifyGetOpArea(unittest.TestCase):
    def setUp(self):
        self.router = default_router.PoliceRouter()
        self.station = self.router._stations.iloc[[8]].geometry.union_all()

    def test_get_operation_area(self):
        """Check that the returned operation area actually contains
        the station"""
        self.assertTrue(self.station.within(
            default_router.get_operation_area(self.station)
        ))

class TestHotareaMethods(unittest.TestCase):
    def setUp(self):
        downscale_factor = 1000
        self.hotarea = HQ.Hotarea(0.6, transform(Polygon(
            ((0,0), (1,8), (9, 7), (8, -2), (0,0))
        ), lambda x: x / downscale_factor))

    def test_hotspots(self):
        """Check that the returned points are all within the area"""
        self.assertTrue(all(
                (point.probability == self.hotarea.probability)
            and (point.pt.within(self.hotarea.area))
                for point in self.hotarea.hotspots()
        ))

class TestRouteComputer(unittest.TestCase):
    def setUp(self):
        self.router = default_router.PoliceRouter()
        self.cai_ids = range(len(self.router._stations.index))

    def test_compute_routes(self):
        """Check that no exceptions arise from the algorithm as a whole"""
        for id in self.cai_ids:
            with self.subTest(id=id):
                self.router.compute_routes(id, 1, include_station=False)

if __name__ == '__main__':
    unittest.main()

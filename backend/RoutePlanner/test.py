from default_router import LonLat, classify_points
import matplotlib.pyplot as plt
import math
from random import uniform

MIN_LON = -5.0
MAX_LON = 5.0

MIN_LAT = -5.0
MAX_LAT = 5.0


def view_classify(n_points, n_cats):
    spots = [LonLat(
        uniform(MIN_LON, MAX_LON),
        uniform(MIN_LAT, MAX_LAT)
    ) for _ in range(n_points)]

    categories = classify_points(spots, n_cats, LonLat(0,0))
    for i in range(len(categories)):
        plt.plot(
            [point.lon for point in categories[i]],
            [point.lat for point in categories[i]],
            "o"
        )
        plt.plot(
            [0, MAX_LON*math.cos(2*i*math.pi/n_cats)],
            [0, MAX_LAT*math.sin(2*i*math.pi/n_cats)],
            color="black"
        )
    plt.show()

if __name__ == "__main__":
    view_classify(200, 5)

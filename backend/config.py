import os

ORS_KEY = os.getenv("ORS_KEY")
if ORS_KEY is None:
    raise EnvironmentError("ORS_KEY not defined in env.")
MODEL_PATH = "crime_model_simple.pkl"
DB_NAME = "route_patrol"
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASSWORD = None

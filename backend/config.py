import os

ORS_KEY = os.getenv("ORS_KEY")
if ORS_KEY is None:
    raise EnvironmentError("ORS_KEY not defined in env.")
MODEL_PATH = "crime_model_simple.pkl"
DB_NAME = "route_patrol"
DB_HOST = "db"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
if DB_PASSWORD is None:
    raise EnvironmentError("DB_PASSWORD not defined in env.")
DB_AUTOPOPULATE = True

# Patrol Routes Backend

This is the backend component of the Patrol Routes application, written
in Python Flask.

## Running

To run it, you will need:

1. An active PostgreSQL database, so please install PostgreSQL and create
   the database `route_patrol`. Make sure the user `postgres` is able
   to access it. If you are using other username, alter
   `database/models.py` accordingly. 
2. An ORS API key and an app key, both of which are set on `.env`.
   Therefore, your `.env` file (inside the project's) backend directory
   should look like:

```
APP_SECRET_KEY="your super secret app key"
ORS_KEY="Your ORS API Key"
```

When you have that set up:

```
cd backend/
poetry install --no-root # install dependencies
poetry run python database/models.py # create tables in the database
poetry run flask run --debug # run the application
```

The app will then run in `http://127.0.0.1:5000/` by default.

## Documentation

The API documentation is automatically generated in `/apidocs` when
running the application, all thanks to `flasgger`. Enjoy!

# Patrol Routes Backend

This is the backend component of the Patrol Routes application, written
in Python Flask.

## Dependencies:
- `flask`
- `flask_cors`
- `flask-restful`
- `flasgger` (documentation)
- `GeoPandas`
- `shapely`
- `openrouteservice`
- `numpy`

## Running

Install the aforementioned dependencies, then:

```
cd backend/
python app.py
```

The app will then run in `http://127.0.0.1:5000/` by default.

You can also install the dependencies autmatically with `poetry`, on
which case you should explicitely run the application with it:

```
cd backend/
poetry install --no-root
poetry run python app.py
```

## Documentation

The API documentation is automatically generated in `/apidocs` when
running the application, all thanks to `flasgger`. Enjoy!

# Heatmap's Openlayer visualization

A very simple application demonstrating the use of OpenLayer's Heatmap feature to view the crime density in the city of Medellín.

## Running

Place the csv in the top directory of the repository. The python conversion script will convert it to json:

```
$ cd experiment/
$ python convert.py
```

Install Node dependencies:
```
$ npm install
```

Run with
```
$ npm start
```

And point your favorite browser to `localhost:5173`.
The dataset is pretty large, so it loading will take a while.

import './style.css';
import data from './hurto_a_persona.json';
import {fromLonLat} from 'ol/proj.js';
import {Map, View} from 'ol';
import HeatmapLayer from 'ol/layer/Heatmap.js';
import Feature from 'ol/Feature.js';
import VectorSource from 'ol/source/Vector.js';
import Point from 'ol/geom/Point.js'
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';

let blur_slider = document.getElementById("blur")
let radius_slider = document.getElementById("radius")

const featurizePoint  = ({latitud, longitud}) => new Feature({
  geometry: new Point(fromLonLat([longitud, latitud])),
});

const vector = new HeatmapLayer({
    opacity: 0.5,
    radius: parseInt(radius_slider.value, 10),
    blur: parseInt(blur_slider.value, 10),
    source: new VectorSource({
        features: data.filter(p => p.longitud && p.latitud)
            .map(featurizePoint),
    })
});

const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM()
    }),
    vector
  ],
  view: new View({
    center: fromLonLat([-75.59055, 6.230833]),
    zoom: 13
  })
});

blur_slider.addEventListener('input', () => {
    console.log("blur: " + blur_slider.value)
    vector.setBlur(parseInt(blur_slider.value, 10))})
radius_slider.addEventListener('input', () => {
    console.log("radius: " + radius_slider.value)
    vector.setRadius(parseInt(radius_slider.value, 10))})

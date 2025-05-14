import { useState } from "react";
import "../App.css";
import {
  Popup,
  Polyline,
  Marker,
  TileLayer,
  Tooltip,
  MapContainer,
  useMap,
  GeoJSON,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import colormap from "colormap";

const NUM_P_COLORS = 20
const COLOR_SCALER = 3

const initial_center = [6.24938, -75.56];
const rev = (pos) => [pos[1], pos[0]];
const probability_colors = [
  "#FAFAFA",
  "#F5F5F5",
  "#EEEEEE",
  "#E0E0E0",
  "#BDBDBD",
  "#9E9E9E",
  "#757575",
  "#616161",
  "#424242",
  "#303030",
  "#212121",
  "#1A1A1A",
  "#121212",
  "#0D0D0D",
  "#080808",
  "#050505",
  "#030303",
  "#020202",
  "#010101",
  "#000000",
];

/*
 * Different colors for the routes. Should be adjusted according to the
 * selected color palette when #16 is solved.
 */
const colors = ["orange", "red", "green", "yellow"];

const fmt_probability = (probability) =>
  (probability * 100).toPrecision(3) + " %";

const paint_cell = (feature) => {
  const probability = Math.pow(feature.properties.probability, 1/COLOR_SCALER);
  const color =
    probability_colors[Math.floor(probability * (NUM_P_COLORS - 1))];
  return {
    color: color,
    fillColor: color,
    fillOpacity: 0.15,
    weight: 0.5,
  };
};

const assignRoute = (routes, index, to) =>
  routes.map((route, i) =>
    i == index ? { ...route, assigned_to: to } : route
  );

function MapCont({ marginLeft, routeInfo, setRouteInfo }) {
  const probabilityTooltip = (feature, layer) => {
    layer.on({
      mouseover: (e) => {
        layer.bindTooltip(fmt_probability(feature.properties.probability));
        layer.openTooltip();
      },
      mouseout: () => {
        layer.unbindTooltip();
        layer.closeTooltip();
      },
    });
  };

  const setRouteAssigns = (i) =>
    setRouteInfo({
      ...routeInfo,
      routes: assignRoute(
        routeInfo.routes,
        i,
        parseInt(prompt("¿A cuál patrulla asignar ruta?"))
      ),
    });
  console.log(routeInfo);

  return (
    <div style={{ marginLeft }}>
      <MapContainer className="map-container" center={initial_center} zoom={13}>
        <TileLayer
          attribution='&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://api.maptiler.com/maps/streets/{z}/{x}/{y}.png?key=0QXa7JOFYu3UV34Ew0Vx"
        />
        {routeInfo && 
          <GeoJSON
            key={JSON.stringify(routeInfo.hotareas)}
            data={routeInfo.hotareas}
            style={(feature) => ({...paint_cell(feature)})}
            onEachFeature={probabilityTooltip}/>}
        {routeInfo &&
          routeInfo.routes.map((route, i) => (
            <Polyline
              pathOptions={{ color: colors[i % colors.length]}}
              key={"r" + i}
              positions={route.geometry.map((pos) => rev(pos))}
              eventHandlers={{
                click: () => setRouteAssigns(i),
              }}
            >
              {route.assigned_to && (
                <Tooltip permanent>
                  {"Asignado a: " + route.assigned_to}
                </Tooltip>
              )}
            </Polyline>
          ))}
        {routeInfo && routeInfo.hotspots.map((spot, index) => 
          <Marker key={"p-" + index} position={rev(spot.coordinates)}/>
        )}
      </MapContainer>
    </div>
  );
}

export default MapCont;

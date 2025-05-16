import { useAuth } from "../Auth";
import { useState, useEffect } from "react";
import "../App.css";
import {
  Popup,
  Polyline,
  Marker,
  Circle,
  TileLayer,
  Tooltip,
  MapContainer,
  useMap,
  GeoJSON,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-polylinedecorator";
import colormap from "colormap";

import { features as stations } from "../../../../geodata/police.json";

const NUM_P_COLORS = 20;
const COLOR_SCALER = 3;

const initial_center = [6.24938, -75.56];
const rev = (pos) => [pos[1], pos[0]];
const off = (pos, i) => pos.map((p) => p + (i - 1) * 0.00002);
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
const colors = ["#E88A4C", "green", "#083D77"];

const fmt_probability = (probability) =>
  (probability * 100).toPrecision(3) + " %";

const paint_cell = (feature) => {
  const probability = Math.pow(
    feature.properties.probability,
    1 / COLOR_SCALER
  );
  const color =
    probability_colors[Math.floor(probability * (NUM_P_COLORS - 1))];
  return {
    color: color,
    fillColor: color,
    fillOpacity: 0.15,
    weight: 0.5,
  };
};

// Custom start marker icon SVG
const startMarkerIcon = L.divIcon({
  html: `<svg width="14" height="14" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="43" fill="#5cb85c" stroke="white" stroke-width="5"/>
    <text x="50" y="64" font-size="40" text-anchor="middle" fill="white">C</text>
  </svg>`,
  className: "",
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

const assignRoute = (routes, index, to) =>
  routes.map((route, i) =>
    i == index ? { ...route, assigned_to: to } : route
  );

function MapCont({ marginLeft, routeInfo, setRouteInfo, selCai }) {
  const { token, user } = useAuth();
  const [map, setMap] = useState(null);

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

  // Add directional arrows to routes when routeInfo changes
  useEffect(() => {
    if (!map || !routeInfo || !routeInfo.routes) return;

    // Remove existing decorators
    map.eachLayer((layer) => {
      if (layer._decorator) {
        map.removeLayer(layer);
      }
    });

    // Add arrow decorators for each route
    routeInfo.routes.forEach((route, i) => {
      if (route.geometry && route.geometry.length > 1) {
        const routePoints = route.geometry.map((pos) => off(rev(pos), i));

        // Create a hidden polyline for the decorator
        const polyline = L.polyline(routePoints, {
          opacity: 0,
          weight: 0,
        }).addTo(map);

        // Create the decorator with arrow pattern
        const decorator = L.polylineDecorator(polyline, {
          patterns: [
            {
              offset: 25,
              repeat: 75,
              symbol: L.Symbol.arrowHead({
                pixelSize: 15,
                headAngle: 50,
                pathOptions: {
                  fillOpacity: 1,
                  weight: 2,
                  color: colors[i % colors.length],
                  fillColor: colors[i % colors.length],
                },
              }),
            },
          ],
        }).addTo(map);

        // Mark this layer for future cleanup
        decorator._decorator = true;
      }
    });
  }, [map, routeInfo]);

  return (
    <div style={{ marginLeft }}>
      <MapContainer
        className="map-container"
        center={initial_center}
        zoom={13}
        whenCreated={setMap}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://api.maptiler.com/maps/streets/{z}/{x}/{y}.png?key=0QXa7JOFYu3UV34Ew0Vx"
        />
        {routeInfo && (
          <GeoJSON
            key={JSON.stringify(routeInfo.hotareas)}
            data={routeInfo.hotareas}
            style={(feature) => ({ ...paint_cell(feature) })}
            onEachFeature={probabilityTooltip}
          />
        )}
        {routeInfo &&
              <Marker
                key={`start-marker`}
                position={rev(stations[selCai].geometry.coordinates)}
                icon={startMarkerIcon}
              ></Marker>}
        {routeInfo &&
          routeInfo.routes.map((route, i) => (
            <>
              {/* Start marker for each route */}
              <Polyline
                pathOptions={{ color: colors[i % colors.length] }}
                key={"r" + i}
                positions={route.geometry.map((pos) => off(rev(pos), i))}
                eventHandlers={user ? {
                  click: () => setRouteAssigns(i),
                } : {}}
              >
                {route.assigned_to && (
                  <Tooltip permanent>
                    {"Asignado a: " + route.assigned_to}
                  </Tooltip>
                )}
              </Polyline>
            </>
          ))}
        {routeInfo && routeInfo.hotspots &&
          routeInfo.hotspots.map((spot, index) => (
            <Circle
              key={`p-${index}`}
              center={rev(spot.coordinates)}
              radius={15}
              fillOpacity={0.6}
              fillColor="#d9534f"
              color="#d9534f"
            >
              <Tooltip>Punto de alto riesgo</Tooltip>
            </Circle>
          ))}
      </MapContainer>
    </div>
  );
}

export default MapCont;

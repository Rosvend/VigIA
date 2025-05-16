import { useState, useEffect, useRef } from "react";
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
const colors = ["orange", "red", "green", "yellow"];

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

// Custom marker icon SVG for CAI stations
const caiMarkerIcon = L.divIcon({
  html: `<svg width="14" height="14" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="50" cy="50" r="40" fill="#5cb85c" stroke="white" stroke-width="4"/>
    <text x="50" y="55" font-size="40" text-anchor="middle" fill="white">C</text>
  </svg>`,
  className: "",
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// A function that adds arrow decorators to a polyline
const addArrowsToPolyline = (map, polylinePoints, color) => {
  // First, make sure the polyline has at least 2 points
  if (!polylinePoints || polylinePoints.length < 2) return null;

  // Create the polyline
  const polyline = L.polyline(polylinePoints, {
    color: color,
    weight: 3,
  }).addTo(map);

  // Create the decorator with arrow patterns
  const decorator = L.polylineDecorator(polyline, {
    patterns: [
      {
        offset: "5%",
        repeat: "15%",
        symbol: L.Symbol.arrowHead({
          pixelSize: 15,
          headAngle: 30,
          polygon: false,
          pathOptions: {
            color: color,
            fillOpacity: 1,
            weight: 3,
          },
        }),
      },
    ],
  }).addTo(map);

  return { polyline, decorator };
};

const assignRoute = (routes, index, to) =>
  routes.map((route, i) =>
    i == index ? { ...route, assigned_to: to } : route
  );

function MapCont({ marginLeft, routeInfo, setRouteInfo }) {
  const [map, setMap] = useState(null);
  const arrowsRef = useRef([]);

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

  const setRouteAssigns = (i) => {
    const patrolNumber = prompt("¿A cuál patrulla asignar ruta?");
    if (patrolNumber && !isNaN(parseInt(patrolNumber))) {
      setRouteInfo({
        ...routeInfo,
        routes: assignRoute(routeInfo.routes, i, parseInt(patrolNumber)),
      });
    }
  };

  // Add directional arrows to routes when routeInfo changes
  useEffect(() => {
    if (!map || !routeInfo || !routeInfo.routes) return;

    // Clean up previous arrows
    if (arrowsRef.current.length > 0) {
      arrowsRef.current.forEach((arrow) => {
        if (arrow.polyline) map.removeLayer(arrow.polyline);
        if (arrow.decorator) map.removeLayer(arrow.decorator);
      });
      arrowsRef.current = [];
    }

    // Add new arrows for each route
    const newArrows = routeInfo.routes
      .map((route, i) => {
        if (route.geometry && route.geometry.length > 1) {
          const routePoints = route.geometry.map((pos) => off(rev(pos), i));
          return addArrowsToPolyline(
            map,
            routePoints,
            colors[i % colors.length]
          );
        }
        return null;
      })
      .filter(Boolean);

    arrowsRef.current = newArrows;

    // Cleanup function
    return () => {
      if (arrowsRef.current.length > 0) {
        arrowsRef.current.forEach((arrow) => {
          if (arrow.polyline) map.removeLayer(arrow.polyline);
          if (arrow.decorator) map.removeLayer(arrow.decorator);
        });
        arrowsRef.current = [];
      }
    };
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
          routeInfo.routes.map((route, i) => (
            <div key={`route-container-${i}`}>
              {/* CAI marker for each route */}
              <Marker
                key={`cai-marker-${i}`}
                position={off(rev(route.geometry[0]), i)}
                icon={caiMarkerIcon}
              >
                <Tooltip permanent>CAI {i + 1}</Tooltip>
              </Marker>

              {/* We're not rendering Polylines here because we handle them with our custom arrows in the useEffect */}

              {/* Display assigned patrol info */}
              {route.assigned_to && (
                <Marker
                  key={`assigned-marker-${i}`}
                  position={off(
                    rev(route.geometry[Math.floor(route.geometry.length / 2)]),
                    i
                  )}
                  icon={L.divIcon({
                    html: `<div style="background-color:${
                      colors[i % colors.length]
                    }; padding: 2px 6px; border-radius: 3px; color: white; font-weight: bold;">Patrulla ${
                      route.assigned_to
                    }</div>`,
                    className: "",
                  })}
                  eventHandlers={{
                    click: () => setRouteAssigns(i),
                  }}
                />
              )}
            </div>
          ))}
        {routeInfo &&
          routeInfo.hotspots.map((spot, index) => (
            <Circle
              key={`p-${index}`}
              center={rev(spot.coordinates)}
              radius={20} // Make circles more visible
              fillOpacity={0.8}
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

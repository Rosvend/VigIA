import { useState } from "react";
import "../App.css";
import {
  Popup,
  Polyline,
  Marker,
  TileLayer,
  MapContainer,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const initial_center = [6.24938, -75.56];
const rev = (pos) => [pos[1], pos[0]];

/*
 * Different colors for the routes. Should be adjusted according to the
 * selected color palette when #16 is solved.
 */
const colors = ["blue", "red", "green", "yellow", "orange", "magenta"];

function MapCont({ marginLeft, routeInfo }) {
  console.log(routeInfo);
  return (
    <div style={{ marginLeft }}>
      <MapContainer className="map-container" center={initial_center} zoom={13}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {routeInfo &&
          routeInfo.hotspots.map(({ coordinates, probability }, i) => (
            <Marker key={"m" + i} position={rev(coordinates)}>
              <Popup>{probability}</Popup>
            </Marker>
          ))}
        {routeInfo &&
          routeInfo.routes.map((route, i) => (
            <Polyline
              pathOptions={{ color: colors[i % colors.length] }}
              key={"r" + i}
              positions={route.map((pos) => rev(pos))}
            />
          ))}
      </MapContainer>
    </div>
  );
}

export default MapCont;

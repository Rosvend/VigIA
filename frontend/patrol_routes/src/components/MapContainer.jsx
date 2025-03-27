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
    <div id="map-container" style={{ marginLeft }}>
      <div id="map">
        {/* Replace with actual map implementation */}
        <img
          src="/assets/MAP_DEMOSTRATION.png"
          alt="Mapa de rutas de patrulla"
        />
      </div>
    </div>
  );
}

export default MapCont;

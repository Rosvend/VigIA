import { useState } from 'react'
import "../App.css"
import { Popup, Marker, TileLayer, MapContainer, useMap } from "react-leaflet"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

const initial_center = [6.24938, -75.56]

function MapCont({ marginLeft }) {
  return (
    <div style={{marginLeft}}>
      <MapContainer className="map-container" center={initial_center} zoom={13}>
  <TileLayer
    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  />
</MapContainer>

    </div>
  );
}

export default MapCont;

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
  GeoJSON
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import colormap from 'colormap'

const NUM_P_COLORS = 20

const initial_center = [6.24938, -75.56];
const rev = (pos) => [pos[1], pos[0]];
const probability_colors = colormap({
  colormap: 'jet',
  nshades: NUM_P_COLORS,
  format: 'hex',
  alpha: 1
})

console.log(probability_colors)

/*
 * Different colors for the routes. Should be adjusted according to the
 * selected color palette when #16 is solved.
 */
const colors = ["blue", "red", "green", "yellow", "orange", "magenta"];

const fmt_probability = (probability) => (probability*100).toPrecision(3) + " %"

const paint_cell = (feature) => {
  const color = probability_colors[Math.round(feature.properties.probability*NUM_P_COLORS)]
  return {color: color}
}

function MapCont({ marginLeft, routeInfo }) {
  console.log(routeInfo);
  const probabilityTooltip = (feature, layer) => {
    layer.on({
      'mouseover': e => {
        layer.bindTooltip(fmt_probability(feature.properties.probability))
        layer.openTooltip()
      },
      'mouseout': () => {
        layer.unbindTooltip()
        layer.closeTooltip()
      }
    })
  }
  return (
    <div style={{ marginLeft }}>
      <MapContainer className="map-container" center={initial_center} zoom={13}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {routeInfo && 
          <GeoJSON
            key={JSON.stringify(routeInfo.hotareas)}
            data={routeInfo.hotareas}
            style={paint_cell}
            onEachFeature={probabilityTooltip}/>}
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

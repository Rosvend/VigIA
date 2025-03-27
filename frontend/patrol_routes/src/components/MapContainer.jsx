function MapContainer({ marginLeft }) {
  return (
    <div id="map-container" style={{ marginLeft }}>
      <div id="map">
        {/*map implementation*/}
        <img
          src="/assets/MAP_DEMOSTRATION.png"
          alt="Mapa de rutas de patrulla"
        />
      </div>
    </div>
  );
}

export default MapContainer;

import { useState } from "react";
import SidebarSection from "./SidebarSection";

function Sidebar({ active }) {
  const [routeCounter, setRouteCounter] = useState(1);
  const [routes, setRoutes] = useState([1]);

  const addRoute = () => {
    const newRouteNumber = routeCounter + 1;
    setRouteCounter(newRouteNumber);
    setRoutes([...routes, newRouteNumber]);
  };

  // Filter section content
  const filterContent = (
    <>
      <div className="form-group">
        <label htmlFor="comunas">Comunas</label>
        <input
          type="text"
          id="comunas"
          className="form-control"
          placeholder="Laureles-Estadio"
        />
      </div>
      <div className="form-group">
        <label htmlFor="crimen">Crimen</label>
        <select id="crimen" className="form-control">
          <option value="hurto">HURTO A MANO ARMADA</option>
          <option value="robo">ROBO</option>
          <option value="homicidio">HOMICIDIO</option>
        </select>
      </div>
      <div className="form-group">
        <label htmlFor="cai">CAI</label>
        <select id="cai" className="form-control">
          <option value="cai1">ESTACION CIRA 64</option>
          <option value="cai2">ESTACION CENTRAL</option>
          <option value="cai3">ESTACION SUR</option>
        </select>
      </div>
      <div className="form-group">
        <label htmlFor="rutas">Rutas Activas</label>
        <div style={{ display: "flex", gap: "10px" }}>
          <select id="rutas" className="form-control form-control-inline">
            {routes.map((route) => (
              <option key={route} value={route}>
                {route}
              </option>
            ))}
          </select>
          <button className="btn" onClick={addRoute}>
            +
          </button>
        </div>
      </div>
    </>
  );

  // Legend section content
  const legendContent = (
    <>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#5cb85c" }}>
          &#9679;
        </div>
        <span>CAI</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#d9534f" }}>
          &#9679;
        </div>
        <span>Punto de Alto Riesgo</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon">&#10092;</div>
        <span>Sentido de la ruta</span>
      </div>
    </>
  );

  return (
    <div className={`sidebar ${active ? "active" : ""}`}>
      <SidebarSection
        title="Filtros"
        content={filterContent}
        defaultActive={true}
      />
      <SidebarSection title="Leyenda" content={legendContent} />
    </div>
  );
}

export default Sidebar;

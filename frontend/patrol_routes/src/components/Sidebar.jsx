import { useState, useEffect } from "react";
import SidebarSection from "./SidebarSection";
import GeoSelect from "./GeoSelect";
import { API_URL } from "../api"

import { features as comunas } from "../../../../geodata/comunas.json"
import { features as stations } from "../../../../geodata/police.json"


function Sidebar({ active, setRouteInfo }) {
  const [selComuna, setSelComuna] = useState(0);
  const [selCai, setSelCai] = useState(0);
  const [routeCounter, setRouteCounter] = useState(1);
  const [routes, setRoutes] = useState([1]);

  const fetchRouteInfo = async (cai, n_routes) => {
    const params = new URLSearchParams();
    params.append("cai", cai);
    params.append("n", n_routes);

    try {
      const response = await fetch(`${API_URL}/routes?${params}`)
      if (response.ok)
        setRouteInfo(await response.json())
    } catch (e) {
      console.log(e)
    }
  }

  useEffect(() => { fetchRouteInfo(selCai, routeCounter) }
    , [selCai, routeCounter]);

  const addRoute = () => {
    const newRouteNumber = routeCounter + 1;
    setRouteCounter(newRouteNumber);
    setRoutes([...routes, newRouteNumber]);
  };

  // Filter section content
  const filterContent = (
    <>
      <GeoSelect features={comunas}
                 selIndex={selComuna}
                 setSelIndex={setSelComuna}
                 name="Comuna"/>
      <div className="form-group">
        <label htmlFor="crimen">Crimen</label>
        <select id="crimen" className="form-control">
          <option value="hurto">HURTO A MANO ARMADA</option>
          <option value="robo">ROBO</option>
          <option value="homicidio">HOMICIDIO</option>
        </select>
      </div>
      <GeoSelect features={stations}
                 selIndex={selCai}
                 setSelIndex={setSelCai}
                 name="Cai"/>
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

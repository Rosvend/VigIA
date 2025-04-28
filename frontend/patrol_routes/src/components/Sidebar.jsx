import { useState, useEffect } from "react";
import SidebarSection from "./SidebarSection";
import GeoSelect from "./GeoSelect";
import { API_URL } from "../api";
import { useAuth } from "../Auth";

import { features as comunas } from "../../../../geodata/comunas.json";
import { features as stations } from "../../../../geodata/police.json";

function Sidebar({ active, setRouteInfo }) {
  const { user } = useAuth();
  const [selComuna, setSelComuna] = useState(0);
  const [selCai, setSelCai] = useState(0);
  const [routeCounter, setRouteCounter] = useState(1);
  const [routes, setRoutes] = useState([1]);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const isSupervisor = user !== null;

  const fetchRouteInfo = async (cai, n_routes) => {
    const params = new URLSearchParams();
    params.append("cai", cai);
    params.append("n", n_routes);
    // TODO: add the requested spots feature (already implemented in backend)
    // params.append("requested_spots", JSON.stringify([[lat, lon]]));

    try {
      const response = await fetch(`${API_URL}/routes?${params}`);
      if (response.ok) setRouteInfo(await response.json());
    } catch (e) {
      console.log(e);
    }
  };

  const addRoute = () => {
    if (routes.length < 4) {
      const newRouteNumber = routeCounter + 1;
      setRouteCounter(newRouteNumber);
      setRoutes([...routes, newRouteNumber]);
    } else {
      alert("Máximo de 4 rutas alcanzado");
    }
  };

  const confirmDeleteRoute = () => {
    setShowDeleteConfirm(true);
  };

  const handleDeleteRoute = (confirm) => {
    if (confirm && routes.length > 1) {
      const newRoutes = routes.slice(1);
      setRoutes(newRoutes);
    }
    setShowDeleteConfirm(false);
  };

  // Filter section content
  const filterContent = (
    <>
      <GeoSelect
        features={comunas}
        selIndex={selComuna}
        setSelIndex={setSelComuna}
        name="Comuna"
      />
      <div className="form-group">
        <label htmlFor="crimen">Crimen</label>
        <select id="crimen" className="form-control">
          <option value="hurto">HURTO A MANO ARMADA</option>
          <option value="robo">ROBO</option>
          <option value="homicidio">HOMICIDIO</option>
        </select>
      </div>
      <GeoSelect
        features={stations}
        selIndex={selCai}
        setSelIndex={setSelCai}
        name="Cai"
      />
      <div className="form-group">
        <label htmlFor="rutas">Rutas Activas</label>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <select
              id="rutas"
              className="form-control form-control-inline"
              multiple
              size={Math.min(routes.length, 4)}
              style={{ width: isSupervisor ? "calc(100% - 80px)" : "100%" }}
            >
              {routes.map((route) => (
                <option key={route} value={route}>
                  Ruta {route}
                </option>
              ))}
            </select>
            {isSupervisor && (
              <div
                style={{ display: "flex", flexDirection: "column", gap: "5px" }}
              >
                <button className="btn" onClick={addRoute} title="Agregar Ruta">
                  +
                </button>
                <button
                  className="btn"
                  onClick={confirmDeleteRoute}
                  title="Eliminar Primera Ruta"
                  disabled={routes.length <= 1}
                >
                  -
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      {showDeleteConfirm && (
        <div className="delete-confirm">
          <p>¿Estás seguro de eliminar la ruta?</p>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: "10px",
            }}
          >
            <button
              className="btn"
              onClick={() => handleDeleteRoute(true)}
              style={{ marginRight: "10px" }}
            >
              Sí
            </button>
            <button className="btn" onClick={() => handleDeleteRoute(false)}>
              No
            </button>
          </div>
        </div>
      )}
      {isSupervisor && (
        <button
          className="btn"
          onClick={() => fetchRouteInfo(selCai, routeCounter)}
        >
          Generar rutas
        </button>
      )}
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

import { useState, useEffect } from "react";
import SidebarSection from "./SidebarSection";
import GeoSelect from "./GeoSelect";
import { API_URL } from "../api";
import { useAuth } from "../Auth";
import { useNotification } from "./NotificationProvider";
import * as turf from "turf";

import { features as comunas } from "../../../../geodata/comunas.json";
import { features as stations } from "../../../../geodata/police.json";

const filterCai = (selComuna) =>
  stations
    .map((cai, i) => ({ ...cai, properties: { id: i, ...cai.properties } }))
    .filter((cai) => turf.inside(cai, comunas[selComuna]));

const findMatchingComuna = (cai) =>
  comunas.findIndex((com) => turf.inside(cai, com));

function Sidebar({ active, routeInfo, setRouteInfo, selCai, setSelCai }) {
  const { token, user } = useAuth();
  const { showSuccess, showError, showInfo } = useNotification();

  const [selComuna, setSelComuna] = useState(
    findMatchingComuna(stations[selCai])
  );
  const [routeCounter, setRouteCounter] = useState(1);
  const [routes, setRoutes] = useState([1]);
  const isSupervisor = user !== null;

  const setComuna = (id) => setSelCai(filterCai(id)[0].properties.id);
  useEffect(() => {
    setSelComuna(findMatchingComuna(stations[selCai]));
  }, [selCai]);

  const assignRoute = async (cai, singleRouteGeom, id) => {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const day = currentDate.getDate();

    try {
      const response = await fetch(
        `${API_URL}/routes/${year}-${month}-${day}/${cai}/${id}`,
        {
          method: "PUT",
          mode: "cors",
          headers: {
            "Content-type": "application/json",
            Authorization: "Bearer " + token,
          },
          body: JSON.stringify(singleRouteGeom),
        }
      );

      if (response.ok) {
        showSuccess(`Ruta asignada correctamente a patrulla ${id}`);
        return true;
      } else {
        showError(`Error al asignar ruta a patrulla ${id}`);
        return false;
      }
    } catch (err) {
      showError(`Error de conexión al asignar ruta: ${err.message}`);
      console.log(err);
      return false;
    }
  };

  const assignRoutes = async (cai, routeInfo) => {
    if (!routeInfo || !routeInfo.routes || routeInfo.routes.length === 0) {
      showError("No hay rutas para asignar");
      return;
    }

    let assignedCount = 0;
    let failedCount = 0;

    for (const route of routeInfo.routes) {
      if (!route.assigned_to) {
        showInfo(`Omitiendo ruta no asignada`);
        continue;
      }

      const success = await assignRoute(cai, route.geometry, route.assigned_to);
      if (success) {
        assignedCount++;
      } else {
        failedCount++;
      }
    }

    if (assignedCount > 0 && failedCount === 0) {
      showSuccess(`${assignedCount} rutas asignadas correctamente`);
    } else if (assignedCount > 0 && failedCount > 0) {
      showInfo(`${assignedCount} rutas asignadas, ${failedCount} fallaron`);
    } else if (failedCount > 0) {
      showError(`Error al asignar ${failedCount} rutas`);
    }
  };

  const fetchRoute = async (cai, id) => {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const day = currentDate.getDate();

    const url = isNaN(id)
      ? `${API_URL}/routes/${year}-${month}-${day}/${cai}`
      : `${API_URL}/routes/${year}-${month}-${day}/${cai}/${id}`;

    try {
      const response = await fetch(url);
      if (response.ok) {
        const jsonResponse = await response.json();
        setRouteInfo(
          isNaN(id) ? { routes: jsonResponse } : { routes: [jsonResponse] }
        );
        showSuccess("Rutas cargadas correctamente");
      } else {
        showError("Error al cargar rutas");
      }
    } catch (err) {
      showError(`Error de conexión: ${err.message}`);
      console.log(err);
    }
  };

  const generateRoutes = async (cai, n_routes) => {
    const params = new URLSearchParams();
    params.append("cai", cai);
    params.append("n", n_routes);
    params.append("hotspots", true);
    // TODO: add the requested spots feature (already implemented in backend)
    // params.append("requested_spots", JSON.stringify([[lat, lon]]));

    try {
      const response = await fetch(`${API_URL}/routes?${params}`);
      if (response.ok) {
        const jsonResponse = await response.json();
        setRouteInfo({
          hotareas: jsonResponse.hotareas,
          hotspots: jsonResponse.hotspots,
          routes: jsonResponse.routes.map((route) => ({
            assigned_to: null,
            geometry: route,
          })),
        });
        showSuccess(`${n_routes} rutas generadas correctamente`);
      } else {
        showError("Error al generar rutas");
      }
    } catch (e) {
      showError(`Error de conexión: ${e.message}`);
      console.log(e);
    }
  };

  const addRoute = () => {
    if (routes.length < 3) {
      const newRouteNumber = routeCounter + 1;
      setRouteCounter(newRouteNumber);
      setRoutes([...routes, newRouteNumber]);
      showInfo(`Ruta ${newRouteNumber} añadida`);
    } else {
      showError("Máximo de 3 rutas alcanzado");
    }
  };

  const resetRoutes = () => {
    setRoutes([1]);
    setRouteCounter(1);
    setRouteInfo(null);
    showInfo("Rutas reiniciadas correctamente");
  };

  // Filter section content
  const filterContent = (
    <>
      <GeoSelect
        features={comunas}
        selIndex={selComuna}
        setSelIndex={setComuna}
        name="Comuna"
      />
      <GeoSelect
        features={filterCai(selComuna)}
        selIndex={selCai}
        setSelIndex={setSelCai}
        idInFeature={true}
        name="Cai"
      />
      <div className="form-group">
        <label htmlFor="rutas">Rutas Activas</label>
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <select
            id="rutas"
            className="form-control"
            multiple
            size={Math.min(routes.length, 3)}
          >
            {routes.map((route) => (
              <option key={route} value={route}>
                Ruta {route}
              </option>
            ))}
          </select>

          {isSupervisor && (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: "10px",
              }}
            >
              <button
                className="btn"
                onClick={addRoute}
                title="Agregar Ruta"
                disabled={routes.length >= 4}
              >
                Agregar ruta
              </button>
              <button
                className="btn"
                onClick={resetRoutes}
                title="Reiniciar rutas"
              >
                Reiniciar
              </button>
            </div>
          )}
        </div>
      </div>
      {isSupervisor ? (
        <>
          <button
            className="btn"
            onClick={() => generateRoutes(selCai, routeCounter)}
            style={{ marginTop: "10px", width: "100%" }}
          >
            Generar rutas
          </button>
          <button
            className="btn"
            onClick={() => assignRoutes(selCai, routeInfo)}
            style={{ marginTop: "10px", width: "100%" }}
            disabled={
              !routeInfo || !routeInfo.routes || routeInfo.routes.length === 0
            }
          >
            Asignar rutas
          </button>
        </>
      ) : (
        <>
          <button
            className="btn"
            onClick={() => fetchRoute(selCai)}
            style={{ marginTop: "10px", width: "100%" }}
          >
            Mostrar rutas
          </button>
        </>
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
        <span>CAI (Centro de Atención Inmediata)</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#d9534f" }}>
          &#9679;
        </div>
        <span>Punto de Alto Riesgo</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#E88A4C" }}>
          &#10148;
        </div>
        <span>Ruta 1</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "green" }}>
          &#10148;
        </div>
        <span>Ruta 2</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#083D77" }}>
          &#10148;
        </div>
        <span>Ruta 3</span>
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

import { useState, useEffect } from "react";
import SidebarSection from "./SidebarSection";
import GeoSelect from "./GeoSelect";
import { API_URL } from "../api";
import { useAuth } from "../Auth";
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
  console.log(stations[selCai]);
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

      // TODO: Insert fancy popup here.
      if (response.ok) return false;
    } catch (err) {
      // TODO: Error message
      console.log(err);
    }
  };

  const assignRoutes = async (cai, routeInfo) => {
    routeInfo.routes.forEach(async (route) => {
      console.log(cai);
      console.log(route.geometry);
      console.log(route.assigned_to);
      await assignRoute(cai, route.geometry, route.assigned_to);
    });
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
      }
    } catch (err) {
      // TODO: show error message properly
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
      }
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

  const resetRoutes = () => {
    setRoutes([1]);
    setRouteCounter(1);
    setRouteInfo(null);
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
            size={Math.min(routes.length, 4)}
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
        <span>CAI</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#d9534f" }}>
          &#9679;
        </div>
        <span>Punto de Alto Riesgo</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "#007bff" }}>
          &#11049;
        </div>
        <span>Inicio de ruta</span>
      </div>
      <div className="legend-item">
        <div className="legend-icon" style={{ color: "white" }}>
          &#10148;
        </div>
        <span>Dirección de ruta</span>
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

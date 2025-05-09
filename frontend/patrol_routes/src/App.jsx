import { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import MapCont from "./components/MapContainer";
import AuthProvider from "./Auth";
import "./App.css";

function App() {
  const [sidebarActive, setSidebarActive] = useState(false);
  const [activeRole, setActiveRole] = useState("policia");
  const [routeInfo, setRouteInfo] = useState(null);

  const toggleSidebar = () => {
    setSidebarActive(!sidebarActive);
  };

  const handleRoleChange = (role) => {
    setActiveRole(role);
  };

  return (
    <AuthProvider>
      <div className="app">
        <Header activeRole={activeRole} onRoleChange={handleRoleChange} />
        <div className="container">
          <button
            className={`burger-menu ${sidebarActive ? "active" : ""}`}
            onClick={toggleSidebar}
          >
            <div className="burger-line"></div>
            <div className="burger-line"></div>
            <div className="burger-line"></div>
          </button>

          <Sidebar setRouteInfo={setRouteInfo} active={sidebarActive} />
          <MapCont
            marginLeft={sidebarActive ? "300px" : "30px"}
            activeRole={activeRole}
            routeInfo={routeInfo}
          />
        </div>
      </div>
    </AuthProvider>
  );
}

export default App;

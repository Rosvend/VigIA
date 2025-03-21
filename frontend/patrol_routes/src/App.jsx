import { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import MapContainer from "./components/MapContainer";
import "./App.css";

function App() {
  const [sidebarActive, setSidebarActive] = useState(false);
  const [activeRole, setActiveRole] = useState("policia");

  const toggleSidebar = () => {
    setSidebarActive(!sidebarActive);
  };

  const handleRoleChange = (role) => {
    setActiveRole(role);
  };

  return (
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

        <Sidebar active={sidebarActive} />
        <MapContainer
          marginLeft={sidebarActive ? "300px" : "0"}
          activeRole={activeRole}
        />
      </div>
    </div>
  );
}

export default App;

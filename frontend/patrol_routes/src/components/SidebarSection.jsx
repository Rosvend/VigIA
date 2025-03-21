import { useState } from "react";

function SidebarSection({ title, content, defaultActive = false }) {
  const [isActive, setIsActive] = useState(defaultActive);

  const toggleActive = () => {
    setIsActive(!isActive);
  };

  return (
    <div className="sidebar-section">
      <div className="sidebar-header" onClick={toggleActive}>
        <h2>{title}</h2>
        <span className={`toggle-arrow ${isActive ? "up" : ""}`}></span>
      </div>
      <div className={`sidebar-content ${isActive ? "active" : ""}`}>
        {content}
      </div>
    </div>
  );
}

export default SidebarSection;

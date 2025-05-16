import { useState } from "react";
import { useAuth } from "../Auth";
import { useNotification } from "./NotificationProvider";
import LoginModal from "./LoginModal";

function Header({
  activeRole,
  onRoleChange,
  sidebarActive,
  toggleSidebar,
  selCai,
  setSelCai,
}) {
  const { user, logIn, logOut } = useAuth();
  const { showSuccess, showError } = useNotification();
  const [showLoginModal, setShowLoginModal] = useState(false);

  const toggleRole = () => {
    if (activeRole === "policia") {
      if (user) {
        onRoleChange("supervisor");
        showSuccess("Modo supervisor activado");
      } else {
        setShowLoginModal(true);
      }
    } else {
      logOut();
      onRoleChange("policia");
      showSuccess("Modo policía activado");
    }
  };

  const handleLogin = (username, password) => {
    logIn(username, password)
      .then(() => {
        setShowLoginModal(false);
        onRoleChange("supervisor");
      })
      .catch(() => {
        // Error is already handled in Auth context
      });
  };

  const handleCloseModal = () => {
    setShowLoginModal(false);
  };

  const goToMyCai = () => {
    if (user && user.cai_id !== undefined && user.cai_id !== selCai) {
      setSelCai(user.cai_id);
      showSuccess(`Navegando a tu CAI asignado`);
    } else {
      showError("No se pudo navegar al CAI asignado");
    }
  };

  return (
    <header>
      <div className="header-content">
        <div className="header-left">
          <button
            className={`burger-menu ${sidebarActive ? "active" : ""}`}
            onClick={toggleSidebar}
          >
            <div className="burger-line"></div>
            <div className="burger-line"></div>
            <div className="burger-line"></div>
          </button>
          <h1>Patrol Routes</h1>
        </div>
        {user && user.cai_id !== undefined && user.cai_id != selCai && (
          <div onClick={goToMyCai} className="role-toggle">
            Ir a mi cai
          </div>
        )}
        <div className="role-toggle" onClick={toggleRole}>
          <span className="role-label">Rol:</span>
          <span className="role-value">
            {activeRole === "policia"
              ? "Policía"
              : `Supervisor ${user ? `(${user.cedula})` : ""}`}
          </span>
          <span className="role-arrow">👮🏻‍♂️</span>
        </div>
      </div>
      {showLoginModal && (
        <LoginModal onLogin={handleLogin} onClose={handleCloseModal} />
      )}
    </header>
  );
}

export default Header;

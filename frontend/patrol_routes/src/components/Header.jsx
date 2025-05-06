import { useState } from "react";
import { useAuth } from "../Auth";
import LoginModal from "./LoginModal";

function Header({ activeRole, onRoleChange }) {
  const { user, logIn, logOut } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);

  const toggleRole = () => {
    if (activeRole === "policia") {
      if (user) {
        onRoleChange("supervisor");
      } else {
        setShowLoginModal(true);
      }
    } else {
      logOut();
      onRoleChange("policia");
    }
  };

  const handleLogin = (username, password) => {
    logIn(username, password).then(() => {
      setShowLoginModal(false);
      onRoleChange("supervisor");
    });
  };

  const handleCloseModal = () => {
    setShowLoginModal(false);
  };

  return (
    <header>
      <div className="header-content">
        <h1>Patrol Routes</h1>
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

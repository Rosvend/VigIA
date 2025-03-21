function Header({ activeRole, onRoleChange }) {
  const toggleRole = () => {
    const newRole = activeRole === "policia" ? "supervisor" : "policia";
    onRoleChange(newRole);
  };

  return (
    <header>
      <div className="header-content">
        <h1>Patrol Routes</h1>
        <div className="role-toggle" onClick={toggleRole}>
          <span className="role-label">Rol:</span>
          <span className="role-value">
            {activeRole === "policia" ? "Policía" : "Supervisor"}
          </span>
          <span className="role-arrow">👮🏻‍♂️</span>
        </div>
      </div>
    </header>
  );
}

export default Header;

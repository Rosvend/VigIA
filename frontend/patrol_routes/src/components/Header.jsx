import {useAuth} from "../Auth"

function Header({ activeRole, onRoleChange }) {
  const {user, logIn} = useAuth();

  const toggleRole = () => {
    const newRole = activeRole === "policia" ? "supervisor" : "policia";
    if (newRole === "supervisor")
      if (!user) {
        // TODO: use a decent login pop-up window instead of plain prompts
        const username = prompt("Cédula")
        const password = prompt("Contraseña")

        logIn(username, password)
      }
    onRoleChange(newRole);
  };

  return (
    <header>
      <div className="header-content">
        <h1>Patrol Routes</h1>
        <div className="role-toggle" onClick={toggleRole}>
          <span className="role-label">Rol:</span>
          <span className="role-value">
            {activeRole === "policia" ? "Policía" : `Supervisor (${user && user.cedula || ""})`}
          </span>
          <span className="role-arrow">👮🏻‍♂️</span>
        </div>
      </div>
    </header>
  );
}

export default Header;

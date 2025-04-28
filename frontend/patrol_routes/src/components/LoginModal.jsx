import { useState } from "react";

function LoginModal({ onLogin, onClose }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username && password) {
      onLogin(username, password);
    }
  };

  return (
    <div className="login-modal-overlay">
      <div className="login-modal">
        <h2>Iniciar Sesión</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="cedula">Cédula</label>
            <input
              type="text"
              id="cedula"
              className="form-control"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Contraseña</label>
            <input
              type="password"
              id="password"
              className="form-control"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div className="modal-buttons">
            <button type="submit" className="btn btn-primary">
              Iniciar sesión
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Cancelar
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginModal;

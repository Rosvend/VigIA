import { useEffect, useContext, useState, createContext } from "react";
import { API_URL } from "./api";
import { useNotification } from "./components/NotificationProvider";

const getStoredUser = () => JSON.parse(localStorage.getItem("user"));

const setStoredUser = (user) =>
  localStorage.setItem("user", JSON.stringify(user));

const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(getStoredUser());
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [error, setError] = useState(null);
  const { showSuccess, showError, showInfo } = useNotification();

  useEffect(() => {
    if (user) {
      setStoredUser(user);
    }
  }, [user]);

  const logIn = async (username, password) => {
    setError(null);
    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    try {
      const response = await fetch(`${API_URL}/admin/login`, {
        method: "POST",
        mode: "cors",
        body: formData,
      });

      if (response.ok) {
        const res = await response.json();
        setUser(res.user);
        setToken(res.access_token);
        localStorage.setItem("token", res.access_token);
        showSuccess(`Inicio de sesión exitoso. Bienvenido, ${username}!`);
        return res;
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.message || "Error al iniciar sesión";
        setError(errorMessage);
        showError(errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      const errorMessage = err.message || "Error de conexión";
      setError(errorMessage);
      showError(errorMessage);
      throw err;
    }
  };

  const logOut = () => {
    setUser(null);
    setToken("");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    showInfo("Sesión cerrada correctamente");
  };

  return (
    <AuthContext.Provider value={{ token, user, logIn, logOut, error }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;

export const useAuth = () => useContext(AuthContext);

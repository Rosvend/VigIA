import { useEffect, useContext, useState, createContext } from "react";
import { API_URL } from "./api";

const getStoredUser = () => JSON.parse(localStorage.getItem("user"));

const setStoredUser = (user) =>
  localStorage.setItem("user", JSON.stringify(user));

const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(getStoredUser());
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [error, setError] = useState(null);

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
        return res;
      } else {
        const errorData = await response.json();
        setError(errorData.message || "Error al iniciar sesión");
        throw new Error(errorData.message || "Error al iniciar sesión");
      }
    } catch (err) {
      setError(err.message || "Error de conexión");
      throw err;
    }
  };

  const logOut = () => {
    setUser(null);
    setToken("");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider value={{ token, user, logIn, logOut, error }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;

export const useAuth = () => useContext(AuthContext);

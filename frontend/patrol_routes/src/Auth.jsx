import { useEffect, useContext, useState, createContext } from "react"
import { API_URL } from "./api"

const getStoredUser = () =>
  JSON.parse(localStorage.getItem("user"));

const setStoredUser = (user) =>
  localStorage.setItem("user", JSON.stringify(user))

const AuthContext = createContext();

const AuthProvider = ({children}) => {
  const [user, setUser] = useState(getStoredUser());
  const [token, setToken] = useState(localStorage.getItem("token") || "")

  useEffect(() => {
    setStoredUser(user);
  }, [user]);

  const logIn = async (username, password) => {
    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(`${API_URL}/admin/login`, {
      method: "POST",
        mode: "cors",
      body: formData
    });
    if (response.ok){
      const res = await response.json();
      setUser(res.user);
      setToken(res.access_token);
      localStorage.setItem("token", res.access_token)
    } else {
      // TODO: handle both invalid credentials response and server error
    }
  };

  const logOut = () => {
    setUser(null);
    setToken("");
    localStorage.removeItem("token")
  };

  return <AuthContext.Provider value={{token, user, logIn, logOut}}>
    {children}
  </AuthContext.Provider>
}

export default AuthProvider;

export const useAuth = () => useContext(AuthContext);

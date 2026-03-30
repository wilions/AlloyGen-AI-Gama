import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { apiLogin, apiRegister } from "../api/auth";
import {
  getToken,
  setToken,
  clearToken,
  getStoredUser,
  setStoredUser,
  type AuthUser,
} from "../api/client";

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (
    email: string,
    password: string,
    displayName?: string
  ) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: getStoredUser(),
    token: getToken(),
    loading: false,
  });

  // Check token validity on mount
  useEffect(() => {
    if (state.token && !state.user) {
      // Token exists but no user — clear stale auth
      clearToken();
      setState({ user: null, token: null, loading: false });
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, loading: true }));
    try {
      const res = await apiLogin(email, password);
      const user: AuthUser = {
        user_id: res.user_id,
        email: res.email,
        display_name: res.display_name,
      };
      setToken(res.access_token);
      setStoredUser(user);
      setState({ user, token: res.access_token, loading: false });
    } catch (err) {
      setState((s) => ({ ...s, loading: false }));
      throw err;
    }
  }, []);

  const register = useCallback(
    async (email: string, password: string, displayName?: string) => {
      setState((s) => ({ ...s, loading: true }));
      try {
        const res = await apiRegister(email, password, displayName);
        const user: AuthUser = {
          user_id: res.user_id,
          email: res.email,
          display_name: res.display_name,
        };
        setToken(res.access_token);
        setStoredUser(user);
        setState({ user, token: res.access_token, loading: false });
      } catch (err) {
        setState((s) => ({ ...s, loading: false }));
        throw err;
      }
    },
    []
  );

  const logout = useCallback(() => {
    clearToken();
    setState({ user: null, token: null, loading: false });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

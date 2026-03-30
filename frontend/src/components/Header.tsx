import { Atom, Sun, Moon, LogOut } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { useAuth } from '../context/AuthContext';
import { useState, useEffect } from 'react';

export function Header() {
  const { state } = useChat();
  const { user, logout } = useAuth();
  const [dark, setDark] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('alloygen_theme') !== 'light';
    }
    return true;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', !dark);
    localStorage.setItem('alloygen_theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <header className="glass rounded-2xl overflow-hidden">
      <div className="px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary/15 flex items-center justify-center">
            <Atom className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-semibold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
              AlloyGen 2.0
            </h1>
            <p className="text-[11px] text-secondary tracking-wide uppercase">
              Intelligent Alloy Design Platform
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5 text-[13px] text-secondary">
            <span
              className={`w-2 h-2 rounded-full transition-all duration-500 ${
                state.wsReady
                  ? 'bg-success shadow-[0_0_6px_var(--color-success)]'
                  : 'bg-warning shadow-[0_0_6px_var(--color-warning)] animate-pulse'
              }`}
            />
            <span className="font-medium">
              {state.wsReady ? 'Connected' : 'Connecting...'}
            </span>
          </div>
          <button
            onClick={() => setDark(d => !d)}
            className="p-1.5 rounded-lg hover:bg-white/10 text-secondary transition-colors"
            title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          {user && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-secondary">{user.display_name || user.email}</span>
              <button
                onClick={logout}
                className="p-1.5 rounded-lg hover:bg-white/10 text-secondary transition-colors"
                title="Log out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
      <div className="h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
    </header>
  );
}

import { Atom } from 'lucide-react';
import { useChat } from '../context/ChatContext';

export function Header() {
  const { state } = useChat();

  return (
    <header className="glass rounded-2xl overflow-hidden">
      <div className="px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary/15 flex items-center justify-center">
            <Atom className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-semibold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
              AlloyGen AI (Gama)
            </h1>
            <p className="text-[11px] text-secondary tracking-wide uppercase">
              Metallurgic & Mechanical Expert
            </p>
          </div>
        </div>
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
      </div>
      <div className="h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
    </header>
  );
}

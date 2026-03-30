import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { Atom } from "lucide-react";

export default function LoginPage() {
  const { login, register, loading } = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (isRegister) {
        await register(email, password, displayName || undefined);
      } else {
        await login(email, password);
      }
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--raw-bg)]">
      {/* Background blobs */}
      <div className="blob blob-1" />
      <div className="blob blob-2" />
      <div className="blob blob-3" />

      <div className="glass p-8 rounded-2xl w-full max-w-md relative z-10">
        <div className="flex items-center justify-center gap-3 mb-6">
          <Atom className="w-8 h-8 text-[var(--raw-primary)]" />
          <h1 className="text-2xl font-bold text-[var(--raw-text-primary)]">
            AlloyGen 2.0
          </h1>
        </div>

        <p className="text-center text-[var(--raw-text-secondary)] mb-6">
          {isRegister ? "Create your account" : "Sign in to continue"}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {isRegister && (
            <input
              type="text"
              placeholder="Display name (optional)"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-[var(--raw-surface-2)] text-[var(--raw-text-primary)] border border-[var(--raw-border)] focus:border-[var(--raw-primary)] focus:outline-none transition-colors"
            />
          )}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full px-4 py-3 rounded-xl bg-[var(--raw-surface-2)] text-[var(--raw-text-primary)] border border-[var(--raw-border)] focus:border-[var(--raw-primary)] focus:outline-none transition-colors"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
            className="w-full px-4 py-3 rounded-xl bg-[var(--raw-surface-2)] text-[var(--raw-text-primary)] border border-[var(--raw-border)] focus:border-[var(--raw-primary)] focus:outline-none transition-colors"
          />

          {error && (
            <p className="text-[var(--raw-error)] text-sm">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl bg-[var(--raw-primary)] text-white font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading
              ? "..."
              : isRegister
              ? "Create Account"
              : "Sign In"}
          </button>
        </form>

        <p className="text-center text-sm text-[var(--raw-text-secondary)] mt-4">
          {isRegister ? "Already have an account?" : "Don't have an account?"}{" "}
          <button
            onClick={() => {
              setIsRegister(!isRegister);
              setError("");
            }}
            className="text-[var(--raw-primary)] hover:underline"
          >
            {isRegister ? "Sign in" : "Register"}
          </button>
        </p>
      </div>
    </div>
  );
}

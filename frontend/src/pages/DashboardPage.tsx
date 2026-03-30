import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { authFetch } from "../api/client";
import {
  Atom,
  Plus,
  MessageSquare,
  Brain,
  Clock,
  LogOut,
} from "lucide-react";

interface SessionSummary {
  id: string;
  state: string;
  targets: string[];
  model_path: string | null;
  created_at: string | null;
  last_active: string | null;
}

interface ModelSummary {
  path: string;
  filename: string;
  best_model_name: string;
  targets: string[];
  score: number;
  feature_count: number;
  timestamp: number;
}

export default function DashboardPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);

  useEffect(() => {
    authFetch("/sessions")
      .then((r) => r.json())
      .then((d) => setSessions(d.sessions || []))
      .catch(() => {});

    fetch("/models")
      .then((r) => r.json())
      .then((d) => setModels(d.models || []))
      .catch(() => {});
  }, []);

  const handleNewSession = () => {
    navigate("/chat");
  };

  const handleContinueSession = (sessionId: string) => {
    navigate(`/chat/${sessionId}`);
  };

  const formatDate = (iso: string | null) => {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="min-h-screen bg-[var(--raw-bg)] text-[var(--raw-text-primary)]">
      {/* Background blobs */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <Atom className="w-8 h-8 text-[var(--raw-primary)]" />
            <h1 className="text-2xl font-bold">AlloyGen 2.0</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-[var(--raw-text-secondary)]">
              {user?.display_name || user?.email}
            </span>
            <button
              onClick={logout}
              className="p-2 rounded-lg hover:bg-[var(--raw-surface-2)] transition-colors"
              title="Sign out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* New Session Button */}
        <button
          onClick={handleNewSession}
          className="glass w-full p-4 rounded-xl flex items-center gap-3 mb-8 hover:border-[var(--raw-primary)] border border-transparent transition-colors"
        >
          <Plus className="w-5 h-5 text-[var(--raw-primary)]" />
          <span className="font-medium">New Session</span>
          <span className="text-sm text-[var(--raw-text-secondary)]">
            Upload a dataset and start predicting
          </span>
        </button>

        {/* Sessions */}
        {sessions.length > 0 && (
          <section className="mb-8">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Recent Sessions
            </h2>
            <div className="grid gap-3">
              {sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleContinueSession(s.id)}
                  className="glass p-4 rounded-xl text-left hover:border-[var(--raw-primary)] border border-transparent transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-medium">
                        {s.targets.length > 0
                          ? `Predicting: ${s.targets.join(", ")}`
                          : "Session"}
                      </span>
                      <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-[var(--raw-surface-2)] text-[var(--raw-text-secondary)]">
                        {s.state}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-[var(--raw-text-secondary)]">
                      <Clock className="w-3 h-3" />
                      {formatDate(s.last_active)}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </section>
        )}

        {/* Models */}
        {models.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Trained Models
            </h2>
            <div className="grid gap-3 md:grid-cols-2">
              {models.map((m) => (
                <div
                  key={m.path}
                  className="glass p-4 rounded-xl"
                >
                  <div className="font-medium text-sm">{m.best_model_name}</div>
                  <div className="text-xs text-[var(--raw-text-secondary)] mt-1">
                    Targets: {m.targets.join(", ")}
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-xs text-[var(--raw-text-secondary)]">
                    <span>Score: {m.score.toFixed(4)}</span>
                    <span>{m.feature_count} features</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {sessions.length === 0 && models.length === 0 && (
          <div className="text-center py-16 text-[var(--raw-text-secondary)]">
            <Atom className="w-12 h-12 mx-auto mb-4 opacity-30" />
            <p>No sessions or models yet. Start a new session to get going!</p>
          </div>
        )}
      </div>
    </div>
  );
}

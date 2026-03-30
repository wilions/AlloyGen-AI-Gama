import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ChatProvider, useChat } from './context/ChatContext';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { ToastContainer } from './components/ToastContainer';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';

function ChatPage() {
  const { state, dispatch } = useChat();
  const { send, sendCancel, sendReset, sendSelectModel } = useWebSocket(state.sessionId, dispatch);

  return (
    <>
      {/* Background blobs */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      <div className="w-[95%] max-w-[1400px] h-[92vh] flex flex-col gap-4 z-10">
        <Header />
        <div className="flex gap-4 flex-1 overflow-hidden">
          <Sidebar send={send} onReset={sendReset} onSelectModel={sendSelectModel} />
          <ChatPanel send={send} sendCancel={sendCancel} />
        </div>
      </div>

      <ToastContainer />
    </>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route
        path="/login"
        element={user ? <Navigate to="/" replace /> : <LoginPage />}
      />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/chat/:sessionId?"
        element={
          <ProtectedRoute>
            <ChatProvider>
              <ChatPage />
            </ChatProvider>
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

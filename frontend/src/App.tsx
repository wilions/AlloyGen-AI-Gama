import { ChatProvider, useChat } from './context/ChatContext';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatPanel } from './components/ChatPanel';
import { ToastContainer } from './components/ToastContainer';

function AppContent() {
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

export default function App() {
  return (
    <ChatProvider>
      <AppContent />
    </ChatProvider>
  );
}

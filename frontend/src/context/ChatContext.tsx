import {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useMemo,
  type ReactNode,
  type Dispatch,
} from 'react';
import type { ChatState, ChatAction, ChatMessage } from '../types';

const STORAGE_KEY = 'alloygen_chat';

// Restore or create session ID
function getSessionId(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.sessionId) return parsed.sessionId;
    }
  } catch {}
  return 'session_' + Math.random().toString(36).substring(2, 11);
}

function getSavedMessages(): ChatMessage[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed.messages)) return parsed.messages;
    }
  } catch {}
  return [];
}

const sessionId = getSessionId();
const savedMessages = getSavedMessages();

const initialState: ChatState = {
  messages: savedMessages,
  currentStep: 'requirements',
  pipelineRunning: false,
  isTyping: false,
  wsReady: false,
  uploadedFilename: null,
  sessionId,
  toasts: [],
};

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] };
    case 'SET_STEP':
      return { ...state, currentStep: action.payload };
    case 'SET_TYPING':
      return { ...state, isTyping: action.payload };
    case 'SET_WS_READY':
      return { ...state, wsReady: action.payload };
    case 'SET_UPLOADED_FILENAME':
      return { ...state, uploadedFilename: action.payload };
    case 'PIPELINE_START':
      return { ...state, pipelineRunning: true };
    case 'PIPELINE_COMPLETE':
      return { ...state, pipelineRunning: false };
    case 'ADD_TOAST':
      return { ...state, toasts: [...state.toasts, action.payload] };
    case 'REMOVE_TOAST':
      return { ...state, toasts: state.toasts.filter((t) => t.id !== action.payload) };
    case 'RESTORE_MESSAGES':
      return { ...state, messages: action.payload };
    case 'RESET':
      return {
        ...initialState,
        sessionId: state.sessionId,
        wsReady: state.wsReady,
        messages: [],
      };
    default:
      return state;
  }
}

const ChatContext = createContext<{
  state: ChatState;
  dispatch: Dispatch<ChatAction>;
} | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Persist messages and sessionId to localStorage
  useEffect(() => {
    try {
      // Only persist non-status messages (keep last 100)
      const persistable = state.messages
        .filter((m) => m.role !== 'status')
        .slice(-100);
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ sessionId: state.sessionId, messages: persistable }),
      );
    } catch {}
  }, [state.messages, state.sessionId]);

  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}

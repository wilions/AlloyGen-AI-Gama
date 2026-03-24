import { useRef, useEffect, useState } from 'react';
import { SendHorizontal, XCircle, Atom, Sparkles, Database, FlaskConical } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { ChatMessage } from './ChatMessage';
import { TypingIndicator } from './TypingIndicator';
import { ReconnectBanner } from './ReconnectBanner';

interface ChatPanelProps {
  send: (msg: string) => void;
  sendCancel: () => void;
}

const SUGGESTIONS = [
  { icon: Sparkles, text: 'Design a high-strength aluminum alloy' },
  { icon: Database, text: 'Upload my dataset for training' },
  { icon: FlaskConical, text: 'Predict properties for a new composition' },
];

function WelcomeState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8 py-12" style={{ animation: 'slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1)' }}>
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/20 flex items-center justify-center">
        <Atom className="w-8 h-8 text-primary" />
      </div>
      <div className="text-center max-w-md">
        <h2 className="text-xl font-semibold text-text-primary mb-2">
          Welcome to AlloyGen AI (Gama)
        </h2>
        <p className="text-sm text-secondary leading-relaxed">
          Your intelligent assistant for alloy design, property prediction, and materials discovery. Start by describing your requirements.
        </p>
      </div>
      <div className="flex flex-wrap gap-3 justify-center mt-2">
        {SUGGESTIONS.map(({ icon: Icon, text }) => (
          <div
            key={text}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-subtle border border-glass-border
              text-[13px] text-secondary cursor-default select-none"
          >
            <Icon className="w-3.5 h-3.5 text-primary shrink-0" />
            {text}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChatPanel({ send, sendCancel }: ChatPanelProps) {
  const { state, dispatch } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages, state.isTyping]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const msg = input.trim();
    if (!msg) return;

    dispatch({
      type: 'ADD_MESSAGE',
      payload: {
        id: crypto.randomUUID(),
        role: 'user',
        content: msg,
        timestamp: Date.now(),
      },
    });
    setInput('');
    dispatch({ type: 'SET_TYPING', payload: true });
    send(msg);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const inputDisabled = !state.wsReady || state.pipelineRunning;
  const placeholder = state.pipelineRunning
    ? 'Pipeline running, please wait...'
    : state.currentStep === 'prediction'
      ? 'Describe an alloy composition for prediction...'
      : 'Discuss requirements or upload a dataset...';

  const hasMessages = state.messages.length > 0;

  return (
    <main className="glass rounded-2xl flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-4 custom-scrollbar">
        {!hasMessages && <WelcomeState />}
        {state.messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {state.isTyping && <TypingIndicator />}
        {!state.wsReady && <ReconnectBanner />}
        <div ref={messagesEndRef} />
      </div>

      <div className="px-6 py-4 border-t border-glass-border bg-surface-overlay rounded-b-2xl">
        {state.pipelineRunning && (
          <div className="flex justify-center mb-3">
            <button
              onClick={sendCancel}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-[13px]
                bg-error/10 border border-error/25 text-error
                hover:bg-error/20 transition-all cursor-pointer"
            >
              <XCircle className="w-4 h-4" />
              Cancel Pipeline
            </button>
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={inputDisabled}
            autoComplete="off"
            rows={1}
            className="chat-textarea flex-1 bg-surface-subtle border border-glass-border rounded-2xl px-5 py-3.5 text-text-primary text-[14px]
              font-[inherit] outline-none transition-all leading-relaxed
              focus:bg-surface-hover focus:border-primary/50 focus:shadow-[0_0_0_3px_var(--color-primary-glow)]
              disabled:opacity-40 disabled:cursor-not-allowed
              placeholder:text-secondary/60"
          />
          <button
            type="submit"
            disabled={inputDisabled || !input.trim()}
            className="w-11 h-11 rounded-xl bg-primary text-white border-none cursor-pointer shrink-0
              flex justify-center items-center transition-all duration-200
              hover:bg-primary-hover hover:shadow-[0_0_16px_var(--color-primary-glow)]
              disabled:bg-secondary/40 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            <SendHorizontal className="w-[18px] h-[18px]" />
          </button>
        </form>
      </div>
    </main>
  );
}

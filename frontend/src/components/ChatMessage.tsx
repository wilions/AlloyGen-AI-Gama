import { lazy, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User } from 'lucide-react';
import type { ChatMessage as ChatMessageType } from '../types';

const ChartRenderer = lazy(() => import('./charts/ChartRenderer'));

interface Props {
  message: ChatMessageType;
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function ChatMessage({ message }: Props) {
  if (message.role === 'status') {
    return (
      <div className="message-fade-in flex justify-center">
        <span className="text-xs text-secondary/70 bg-surface-subtle border border-glass-border rounded-full px-4 py-1.5 inline-flex items-center gap-2">
          <span className="w-1 h-1 rounded-full bg-primary animate-pulse" />
          {message.content}
        </span>
      </div>
    );
  }

  const isUser = message.role === 'user';

  return (
    <div className={`message-fade-in flex gap-3 max-w-[85%] ${isUser ? 'self-end flex-row-reverse' : 'self-start'}`}>
      <div
        className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center mt-1 ${
          isUser
            ? 'bg-primary/15 text-primary'
            : 'bg-accent/15 text-accent'
        }`}
      >
        {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
      </div>
      <div className="flex flex-col gap-1">
        <div
          className={`px-4 py-3 rounded-2xl text-[14px] leading-relaxed ${
            isUser
              ? 'bg-primary/10 border border-primary/15 rounded-tr-sm'
              : 'bg-bot-msg border border-glass-border rounded-tl-sm'
          }`}
        >
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <div className="bot-markdown">
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {message.chartData && (
                <Suspense fallback={<div className="text-gray-500 text-sm py-2">Loading chart...</div>}>
                  <ChartRenderer chartData={message.chartData} />
                </Suspense>
              )}
            </div>
          )}
        </div>
        <span className={`text-[10px] text-secondary/50 px-1 ${isUser ? 'text-right' : ''}`}>
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
}

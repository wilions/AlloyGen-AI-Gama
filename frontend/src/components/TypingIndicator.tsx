import { Bot } from 'lucide-react';

export function TypingIndicator() {
  return (
    <div className="message-fade-in flex gap-3 self-start">
      <div className="w-7 h-7 rounded-lg shrink-0 flex items-center justify-center mt-1 bg-accent/15 text-accent">
        <Bot className="w-3.5 h-3.5" />
      </div>
      <div className="flex gap-1.5 px-4 py-3.5 bg-bot-msg rounded-2xl rounded-tl-sm border border-glass-border w-fit">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-1.5 h-1.5 bg-secondary rounded-full"
            style={{
              animation: 'typing 1.4s infinite ease-in-out both',
              animationDelay: `${-0.32 + i * 0.16}s`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

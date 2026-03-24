import { Loader2 } from 'lucide-react';

export function ReconnectBanner() {
  return (
    <div className="message-fade-in self-center bg-warning/15 border border-warning/40 text-warning rounded-lg px-5 py-2.5 text-sm flex items-center gap-2.5">
      <Loader2 className="w-4 h-4 animate-spin" />
      Connection lost. Reconnecting...
    </div>
  );
}

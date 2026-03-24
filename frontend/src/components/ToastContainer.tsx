import { useEffect } from 'react';
import { X, CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { useChat } from '../context/ChatContext';

const ICONS = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
};

const COLORS = {
  success: 'bg-success/20 border-success/40 text-success',
  error: 'bg-error/20 border-error/40 text-error',
  info: 'bg-primary/20 border-primary/40 text-primary',
};

export function ToastContainer() {
  const { state, dispatch } = useChat();

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {state.toasts.map((toast) => {
        const Icon = ICONS[toast.type];
        return <ToastItem key={toast.id} id={toast.id} icon={Icon} color={COLORS[toast.type]} message={toast.message} dispatch={dispatch} />;
      })}
    </div>
  );
}

function ToastItem({
  id,
  icon: Icon,
  color,
  message,
  dispatch,
}: {
  id: string;
  icon: typeof CheckCircle2;
  color: string;
  message: string;
  dispatch: ReturnType<typeof useChat>['dispatch'];
}) {
  useEffect(() => {
    const timer = setTimeout(() => {
      dispatch({ type: 'REMOVE_TOAST', payload: id });
    }, 5000);
    return () => clearTimeout(timer);
  }, [id, dispatch]);

  return (
    <div
      className={`message-fade-in flex items-center gap-2 px-4 py-3 rounded-lg border text-sm ${color}`}
    >
      <Icon className="w-4 h-4 shrink-0" />
      <span className="flex-1">{message}</span>
      <button
        onClick={() => dispatch({ type: 'REMOVE_TOAST', payload: id })}
        className="opacity-60 hover:opacity-100 transition-opacity"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

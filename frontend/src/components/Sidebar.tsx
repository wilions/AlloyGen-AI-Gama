import { RotateCcw } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { PipelineTracker } from './PipelineTracker';
import { FileUpload } from './FileUpload';
import { ModelSelector } from './ModelSelector';

interface SidebarProps {
  send: (msg: string) => void;
  onReset: () => void;
  onSelectModel: (modelPath: string) => void;
}

export function Sidebar({ send, onReset, onSelectModel }: SidebarProps) {
  const { state } = useChat();
  const canReset = state.currentStep !== 'requirements' || state.pipelineRunning;

  return (
    <aside className="glass rounded-2xl p-5 w-72 flex flex-col gap-1 shrink-0 overflow-y-auto custom-scrollbar">
      <PipelineTracker />

      <div className="h-px bg-glass-border my-4" />

      <FileUpload send={send} />

      <div className="h-px bg-glass-border my-4" />

      <ModelSelector
        onSelect={onSelectModel}
        currentStep={state.currentStep}
        wsReady={state.wsReady}
      />

      {canReset && (
        <>
          <div className="h-px bg-glass-border my-4" />
          <button
            onClick={onReset}
            className="w-full py-2.5 bg-surface-subtle text-secondary border border-glass-border rounded-xl text-[13px] font-medium
              cursor-pointer transition-all flex justify-center items-center gap-2
              hover:bg-surface-hover hover:text-text-primary hover:border-glass-border"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Pipeline
          </button>
        </>
      )}
    </aside>
  );
}

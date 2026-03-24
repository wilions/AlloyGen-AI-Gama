import { CheckCircle2, ListChecks } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { PIPELINE_STEPS } from '../constants';

export function PipelineTracker() {
  const { state } = useChat();
  const currentIndex = PIPELINE_STEPS.findIndex((s) => s.key === state.currentStep);

  return (
    <div>
      <h2 className="text-[13px] font-semibold text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
        <ListChecks className="w-4 h-4 text-primary" />
        ML Pipeline
      </h2>
      <div className="relative">
        {/* Vertical progress line */}
        <div className="absolute left-[15px] top-2 bottom-2 w-px bg-glass-border" />
        <div
          className="absolute left-[15px] top-2 w-px bg-gradient-to-b from-success to-primary transition-all duration-500"
          style={{ height: currentIndex > 0 ? `${(currentIndex / (PIPELINE_STEPS.length - 1)) * 100}%` : '0%' }}
        />

        <ul className="flex flex-col gap-1 relative">
          {PIPELINE_STEPS.map((step, index) => {
            const isCompleted = index < currentIndex;
            const isActive = index === currentIndex;
            const Icon = isCompleted ? CheckCircle2 : step.icon;

            return (
              <li
                key={step.key}
                className={`flex items-center gap-3 text-[13px] px-3 py-2 rounded-lg transition-all duration-300 ${
                  isActive
                    ? 'bg-primary/8 text-primary font-medium'
                    : isCompleted
                      ? 'text-success'
                      : 'text-secondary/60'
                }`}
              >
                <div className={`relative z-10 w-[18px] h-[18px] flex items-center justify-center rounded-full ${
                  isActive ? 'ring-2 ring-primary/30' : ''
                }`}
                  style={isActive ? { animation: 'pulseGlow 2s infinite' } : undefined}
                >
                  <Icon className="w-3.5 h-3.5 shrink-0" />
                </div>
                {step.label}
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

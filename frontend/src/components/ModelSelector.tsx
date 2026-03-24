import { useEffect, useState } from 'react';
import { Brain, ChevronDown, ChevronUp, Clock, Target, CheckCircle2 } from 'lucide-react';
import { fetchModels } from '../lib/api';
import type { ModelInfo } from '../types';

interface ModelSelectorProps {
  onSelect: (modelPath: string) => void;
  currentStep: string;
  wsReady: boolean;
}

function formatDate(ts: number) {
  if (!ts) return 'Unknown';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatScore(score: number, taskType: string) {
  if (taskType === 'classification') return `${(score * 100).toFixed(1)}%`;
  return score.toFixed(4);
}

function scoreLabel(taskType: string) {
  return taskType === 'classification' ? 'Accuracy' : 'R²';
}

export function ModelSelector({ onSelect, currentStep, wsReady }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  // Fetch models when expanded or when training completes (step changes to prediction)
  useEffect(() => {
    if (!expanded && currentStep !== 'prediction') return;
    setLoading(true);
    fetchModels()
      .then((m) => {
        setModels(m);
        // Auto-expand when a new model arrives after training
        if (currentStep === 'prediction' && !expanded && m.length > 0) {
          setExpanded(true);
        }
      })
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [expanded, currentStep]);

  const handleSelect = (model: ModelInfo) => {
    setSelectedPath(model.path);
    onSelect(model.path);
  };

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-[13px] font-semibold text-secondary uppercase tracking-wider cursor-pointer bg-transparent border-none p-0"
      >
        <span className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-primary" />
          Trained Models
        </span>
        {expanded ? (
          <ChevronUp className="w-3.5 h-3.5" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5" />
        )}
      </button>

      {expanded && (
        <div className="mt-3 flex flex-col gap-2" style={{ animation: 'slideUp 0.2s ease-out' }}>
          {loading && (
            <p className="text-[12px] text-secondary text-center py-3">Loading models...</p>
          )}
          {!loading && models.length === 0 && (
            <p className="text-[12px] text-secondary text-center py-3">No trained models found</p>
          )}
          {!loading &&
            models.map((model) => {
              const isSelected = selectedPath === model.path && currentStep === 'prediction';
              return (
                <button
                  key={model.path}
                  onClick={() => handleSelect(model)}
                  disabled={!wsReady}
                  className={`w-full text-left p-3 rounded-xl border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-primary/10 border-primary/30'
                      : 'bg-surface-subtle border-glass-border hover:bg-surface-hover hover:border-glass-border'
                  } disabled:opacity-40 disabled:cursor-not-allowed`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[13px] font-medium text-text-primary flex items-center gap-1.5">
                      {isSelected && <CheckCircle2 className="w-3 h-3 text-success" />}
                      {model.best_model_name}
                    </span>
                    <span className="text-[11px] text-success font-mono">
                      {scoreLabel(model.task_type)} {formatScore(model.score, model.task_type)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-secondary">
                    <Target className="w-3 h-3 shrink-0" />
                    <span className="truncate">{model.targets.join(', ')}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[11px] text-secondary mt-1">
                    <Clock className="w-3 h-3 shrink-0" />
                    {formatDate(model.timestamp)}
                    <span className="ml-auto">{model.feature_count} features</span>
                  </div>
                </button>
              );
            })}
        </div>
      )}
    </div>
  );
}

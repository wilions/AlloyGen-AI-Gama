export type PipelineStep =
  | 'requirements'
  | 'data_prep'
  | 'online_search'
  | 'model_search'
  | 'training'
  | 'prediction';

// Chart data types for inline visualizations
export type ChartData =
  | { type: 'model_comparison'; data: { name: string; score: number; cv_score?: number }[] }
  | { type: 'feature_importance'; data: { name: string; importance: number }[] }
  | { type: 'shap_waterfall'; data: { name: string; value: number }[]; base_value: number }
  | { type: 'prediction_gauge'; data: { target: string; value: number; lower: number; upper: number; confidence: string; train_min: number; train_max: number }[] }
  | { type: 'pareto_front'; data: { candidates: Record<string, number>[]; pareto: Record<string, number>[]; objectives: string[] } }
  | { type: 'correlation_heatmap'; data: { columns: string[]; matrix: number[][] } }
  | { type: 'distribution'; data: { column: string; values: number[]; bins?: number }[] };

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'status';
  content: string;
  timestamp: number;
  chartData?: ChartData;
}

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
}

export interface ModelInfo {
  path: string;
  filename: string;
  best_model_name: string;
  targets: string[];
  task_type: string;
  score: number;
  cv_score: number;
  timestamp: number;
  features: string[];
  feature_count: number;
}

export interface ChatState {
  messages: ChatMessage[];
  currentStep: PipelineStep;
  pipelineRunning: boolean;
  isTyping: boolean;
  wsReady: boolean;
  uploadedFilename: string | null;
  sessionId: string;
  toasts: Toast[];
}

export type ChatAction =
  | { type: 'ADD_MESSAGE'; payload: ChatMessage }
  | { type: 'SET_STEP'; payload: PipelineStep }
  | { type: 'SET_TYPING'; payload: boolean }
  | { type: 'SET_WS_READY'; payload: boolean }
  | { type: 'SET_UPLOADED_FILENAME'; payload: string }
  | { type: 'PIPELINE_START' }
  | { type: 'PIPELINE_COMPLETE' }
  | { type: 'ADD_TOAST'; payload: Toast }
  | { type: 'REMOVE_TOAST'; payload: string }
  | { type: 'RESTORE_MESSAGES'; payload: ChatMessage[] }
  | { type: 'RESET' };

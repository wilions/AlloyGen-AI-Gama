export type PipelineStep =
  | 'requirements'
  | 'data_prep'
  | 'online_search'
  | 'model_search'
  | 'training'
  | 'prediction';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'status';
  content: string;
  timestamp: number;
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

import {
  MessageSquare,
  Sparkles,
  Globe,
  Microscope,
  Dumbbell,
  Wand2,
} from 'lucide-react';
import type { PipelineStep } from './types';

export const PIPELINE_STEPS: {
  key: PipelineStep;
  label: string;
  icon: typeof MessageSquare;
}[] = [
  { key: 'requirements', label: 'Requirements Gathering', icon: MessageSquare },
  { key: 'data_prep', label: 'Data Reorganization', icon: Sparkles },
  { key: 'online_search', label: 'Online Data Search', icon: Globe },
  { key: 'model_search', label: 'Model Search', icon: Microscope },
  { key: 'training', label: 'Training & Selection', icon: Dumbbell },
  { key: 'prediction', label: 'Prediction Ready', icon: Wand2 },
];

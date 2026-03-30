import { lazy, Suspense } from 'react';
import type { ChartData } from '../../types';

// Lazy-load all chart components
const ModelComparisonChart = lazy(() => import('./ModelComparisonChart'));
const FeatureImportanceChart = lazy(() => import('./FeatureImportanceChart'));
const SHAPWaterfall = lazy(() => import('./SHAPWaterfall'));
const PredictionGauge = lazy(() => import('./PredictionGauge'));
const ParetoFrontChart = lazy(() => import('./ParetoFrontChart'));
const CorrelationHeatmap = lazy(() => import('./CorrelationHeatmap'));
const DistributionChart = lazy(() => import('./DistributionChart'));

interface Props {
  chartData: ChartData;
}

export default function ChartRenderer({ chartData }: Props) {
  return (
    <Suspense fallback={<div className="h-32 flex items-center justify-center text-gray-500 text-sm">Loading chart...</div>}>
      {chartData.type === 'model_comparison' && (
        <ModelComparisonChart data={chartData.data} />
      )}
      {chartData.type === 'feature_importance' && (
        <FeatureImportanceChart data={chartData.data} />
      )}
      {chartData.type === 'shap_waterfall' && (
        <SHAPWaterfall data={chartData.data} base_value={chartData.base_value} />
      )}
      {chartData.type === 'prediction_gauge' && (
        <PredictionGauge data={chartData.data} />
      )}
      {chartData.type === 'pareto_front' && (
        <ParetoFrontChart data={chartData.data} />
      )}
      {chartData.type === 'correlation_heatmap' && (
        <CorrelationHeatmap data={chartData.data} />
      )}
      {chartData.type === 'distribution' && (
        <DistributionChart data={chartData.data} />
      )}
    </Suspense>
  );
}

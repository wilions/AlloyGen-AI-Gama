import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ErrorBar, ReferenceLine } from 'recharts';

interface PredictionData {
  target: string;
  value: number;
  lower: number;
  upper: number;
  confidence: string;
  train_min: number;
  train_max: number;
}

interface Props {
  data: PredictionData[];
}

const confidenceColor = (c: string) => {
  switch (c) {
    case 'high': return '#10b981';
    case 'medium': return '#f59e0b';
    case 'low': return '#ef4444';
    default: return '#6b7280';
  }
};

export default function PredictionGauge({ data }: Props) {
  const chartData = data.map(d => ({
    ...d,
    errorLow: d.value - d.lower,
    errorHigh: d.upper - d.value,
    range: d.train_max - d.train_min,
  }));

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Prediction with Confidence Intervals</h4>
      <div className="space-y-3">
        {chartData.map((d, idx) => {
          const pct = d.range > 0 ? ((d.value - d.train_min) / d.range) * 100 : 50;
          const lowerPct = d.range > 0 ? ((d.lower - d.train_min) / d.range) * 100 : 40;
          const upperPct = d.range > 0 ? ((d.upper - d.train_min) / d.range) * 100 : 60;

          return (
            <div key={idx}>
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{d.target}</span>
                <span className="font-mono">
                  {d.value.toFixed(4)}
                  {d.lower !== d.upper && (
                    <span className="text-gray-500"> [{d.lower.toFixed(2)}, {d.upper.toFixed(2)}]</span>
                  )}
                </span>
              </div>
              <div className="relative h-6 rounded bg-gray-800 overflow-hidden">
                {/* Training range background */}
                <div className="absolute inset-0 bg-gray-700/30" />
                {/* Confidence interval */}
                {d.lower !== d.upper && (
                  <div
                    className="absolute h-full opacity-30"
                    style={{
                      left: `${Math.max(0, Math.min(100, lowerPct))}%`,
                      width: `${Math.max(0, Math.min(100, upperPct - lowerPct))}%`,
                      backgroundColor: confidenceColor(d.confidence),
                    }}
                  />
                )}
                {/* Prediction marker */}
                <div
                  className="absolute w-1 h-full"
                  style={{
                    left: `${Math.max(0, Math.min(100, pct))}%`,
                    backgroundColor: confidenceColor(d.confidence),
                  }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                <span>{d.train_min.toFixed(1)}</span>
                <span
                  className="px-1 rounded text-[10px]"
                  style={{ color: confidenceColor(d.confidence) }}
                >
                  {d.confidence} confidence
                </span>
                <span>{d.train_max.toFixed(1)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

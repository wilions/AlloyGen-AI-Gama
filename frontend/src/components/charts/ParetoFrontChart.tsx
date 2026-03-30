import { lazy, Suspense, useState } from 'react';

// Lazy-load Plotly to avoid +3MB initial bundle
const Plot = lazy(() => import('react-plotly.js'));

interface Props {
  data: {
    candidates: Record<string, number>[];
    pareto: Record<string, number>[];
    objectives: string[];
  };
}

export default function ParetoFrontChart({ data }: Props) {
  const { candidates, pareto, objectives } = data;
  const [selected, setSelected] = useState<Record<string, number> | null>(null);

  if (objectives.length < 2) return null;

  const [xKey, yKey] = objectives;

  // Sort pareto front by x for connecting line
  const paretoSorted = [...pareto].sort((a, b) => (a[xKey] ?? 0) - (b[xKey] ?? 0));

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Pareto Front — Multi-Objective Optimization</h4>
      <Suspense fallback={<div className="h-64 flex items-center justify-center text-gray-500">Loading chart...</div>}>
        <Plot
          data={[
            // All candidates
            {
              x: candidates.map(c => c[xKey]),
              y: candidates.map(c => c[yKey]),
              mode: 'markers',
              type: 'scatter',
              name: 'Candidates',
              marker: { color: '#6b7280', size: 5, opacity: 0.4 },
              hovertemplate: `${xKey}: %{x:.4f}<br>${yKey}: %{y:.4f}<extra></extra>`,
            },
            // Pareto front points
            {
              x: pareto.map(c => c[xKey]),
              y: pareto.map(c => c[yKey]),
              mode: 'markers',
              type: 'scatter',
              name: 'Pareto Optimal',
              marker: { color: '#ef4444', size: 10, symbol: 'star' },
              hovertemplate: `${xKey}: %{x:.4f}<br>${yKey}: %{y:.4f}<extra>Pareto</extra>`,
            },
            // Pareto frontier line
            {
              x: paretoSorted.map(c => c[xKey]),
              y: paretoSorted.map(c => c[yKey]),
              mode: 'lines',
              type: 'scatter',
              name: 'Frontier',
              line: { color: '#ef4444', dash: 'dash', width: 1.5 },
              showlegend: false,
            },
          ]}
          layout={{
            width: 500,
            height: 380,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'rgba(31,41,55,0.5)',
            font: { color: '#d1d5db', size: 11 },
            xaxis: { title: xKey, gridcolor: '#374151' },
            yaxis: { title: yKey, gridcolor: '#374151' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.3)' },
            margin: { l: 60, r: 20, t: 20, b: 50 },
            dragmode: 'lasso',
          }}
          config={{ responsive: true, displayModeBar: true }}
          onClick={(event: any) => {
            const point = event.points?.[0];
            if (point && point.curveNumber === 1) {
              setSelected(pareto[point.pointIndex]);
            }
          }}
        />
      </Suspense>
      {selected && (
        <div className="mt-2 p-2 rounded bg-gray-800 text-xs">
          <span className="text-gray-400 font-semibold">Selected candidate: </span>
          {Object.entries(selected).map(([k, v]) => (
            <span key={k} className="mr-3">
              <span className="text-gray-400">{k}:</span>{' '}
              <span className="text-emerald-400 font-mono">{typeof v === 'number' ? v.toFixed(4) : v}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

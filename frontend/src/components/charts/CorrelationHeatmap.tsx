import { lazy, Suspense } from 'react';

const Plot = lazy(() => import('react-plotly.js'));

interface Props {
  data: {
    columns: string[];
    matrix: number[][];
  };
}

export default function CorrelationHeatmap({ data }: Props) {
  const { columns, matrix } = data;

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Feature Correlation Matrix</h4>
      <Suspense fallback={<div className="h-64 flex items-center justify-center text-gray-500">Loading chart...</div>}>
        <Plot
          data={[
            {
              z: matrix,
              x: columns,
              y: columns,
              type: 'heatmap',
              colorscale: 'RdBu',
              zmid: 0,
              zmin: -1,
              zmax: 1,
              hovertemplate: '%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
              colorbar: {
                title: 'r',
                titleside: 'right',
                tickfont: { color: '#9ca3af' },
                titlefont: { color: '#9ca3af' },
              },
            },
          ]}
          layout={{
            width: 500,
            height: 450,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#d1d5db', size: 10 },
            xaxis: { tickangle: -45, tickfont: { size: 9 } },
            yaxis: { tickfont: { size: 9 }, autorange: 'reversed' },
            margin: { l: 80, r: 40, t: 20, b: 80 },
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </Suspense>
    </div>
  );
}

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface Props {
  data: { name: string; score: number; cv_score?: number }[];
}

export default function ModelComparisonChart({ data }: Props) {
  const sorted = [...data]
    .filter(d => d.score != null)
    .sort((a, b) => b.score - a.score);

  const bestScore = sorted[0]?.score ?? 0;

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Model Comparison</h4>
      <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 28)}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 120, right: 20 }}>
          <XAxis type="number" domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#d1d5db', fontSize: 11 }}
            width={110}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#e5e7eb' }}
            formatter={(value: number, name: string) => [value.toFixed(4), name === 'score' ? 'Test Score' : 'CV Score']}
          />
          <Bar dataKey="score" name="Test Score" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.score === bestScore ? '#10b981' : '#6366f1'}
                fillOpacity={entry.score === bestScore ? 1 : 0.7}
              />
            ))}
          </Bar>
          {sorted.some(d => d.cv_score != null) && (
            <Bar dataKey="cv_score" name="CV Score" fill="#f59e0b" fillOpacity={0.5} radius={[0, 4, 4, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

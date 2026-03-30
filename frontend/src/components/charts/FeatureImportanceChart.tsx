import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface Props {
  data: { name: string; importance: number }[];
}

export default function FeatureImportanceChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => b.importance - a.importance).slice(0, 15);
  const maxImp = sorted[0]?.importance ?? 1;

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Feature Importance</h4>
      <ResponsiveContainer width="100%" height={Math.max(180, sorted.length * 26)}>
        <BarChart data={sorted} layout="vertical" margin={{ left: 100, right: 20 }}>
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#d1d5db', fontSize: 11 }}
            width={90}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#e5e7eb' }}
            formatter={(value: number) => [value.toFixed(4), 'Importance']}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, idx) => (
              <Cell
                key={idx}
                fill={`rgba(99, 102, 241, ${0.3 + 0.7 * (entry.importance / maxImp)})`}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

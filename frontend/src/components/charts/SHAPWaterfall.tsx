import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface Props {
  data: { name: string; value: number }[];
  base_value: number;
}

export default function SHAPWaterfall({ data, base_value }: Props) {
  const sorted = [...data]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-1">SHAP Feature Contributions</h4>
      <p className="text-xs text-gray-500 mb-2">
        Base value: {base_value.toFixed(4)} | Red = pushes up, Blue = pushes down
      </p>
      <ResponsiveContainer width="100%" height={Math.max(160, sorted.length * 26)}>
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
            formatter={(value: number) => [
              `${value > 0 ? '+' : ''}${value.toFixed(4)}`,
              'SHAP Value'
            ]}
          />
          <ReferenceLine x={0} stroke="#4b5563" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.value > 0 ? '#ef4444' : '#3b82f6'}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

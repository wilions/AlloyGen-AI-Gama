import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
  data: { column: string; values: number[]; bins?: number }[];
}

function histogram(values: number[], bins: number = 20) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / bins;

  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    counts[idx]++;
  }

  return counts.map((count, i) => ({
    range: `${(min + i * binWidth).toFixed(2)}`,
    count,
  }));
}

export default function DistributionChart({ data }: Props) {
  return (
    <div className="my-3 rounded-lg bg-gray-900/50 p-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Feature Distributions</h4>
      <div className="grid grid-cols-2 gap-3">
        {data.slice(0, 6).map((d, idx) => {
          const histData = histogram(d.values, d.bins ?? 15);
          return (
            <div key={idx}>
              <p className="text-xs text-gray-400 mb-1 truncate">{d.column}</p>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart data={histData} margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
                  <XAxis dataKey="range" tick={false} />
                  <YAxis tick={false} width={0} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 6, fontSize: 11 }}
                    formatter={(value: number) => [value, 'Count']}
                    labelFormatter={(label) => `Value: ${label}`}
                  />
                  <Bar dataKey="count" fill="#6366f1" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}

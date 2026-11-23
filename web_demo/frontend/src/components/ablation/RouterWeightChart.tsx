import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from 'recharts';
import { RouterWeights } from '@/types';

interface RouterWeightChartProps {
  weights: RouterWeights;
}

export const RouterWeightChart: React.FC<RouterWeightChartProps> = ({ weights }) => {
  const data = [
    {
      name: 'DNA Branch',
      value: Number((weights.dna_branch * 100).toFixed(1)),
      color: '#6366f1',
    },
    {
      name: 'Protein Branch',
      value: Number((weights.protein_branch * 100).toFixed(1)),
      color: '#10b981',
    },
    {
      name: 'Bio & Geom Branch',
      value: Number((weights.tabular_branch * 100).toFixed(1)),
      color: '#f59e0b',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-2">
        <h4 className="text-sm font-semibold text-slate-100">
          RB-Net Dynamic Router Allocation
        </h4>
        <p className="text-xs text-slate-400">
          Learned gating weights across active residual expert branches (%)
        </p>
      </div>

      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 42, left: 16, bottom: 8 }}>
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis
              type="category"
              dataKey="name"
              width={125}
              tick={{ fill: '#cbd5e1', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
              contentStyle={{
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '12px',
              }}
              formatter={(val) => [`${Number(val ?? 0)}%`, 'Router Weight']}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={22}>
              {data.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={entry.color} />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v) => `${Number(v ?? 0)}%`}
                style={{ fill: '#e2e8f0', fontSize: 12, fontWeight: 600 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
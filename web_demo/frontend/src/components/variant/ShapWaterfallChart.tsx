import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ShapContribution } from '@/types';

interface ShapWaterfallChartProps {
  explanations: ShapContribution[];
}

export const ShapWaterfallChart: React.FC<ShapWaterfallChartProps> = ({ explanations }) => {
  const sortedData = [...explanations]
    .sort((a, b) => Math.abs(b.shap_value_raw_margin) - Math.abs(a.shap_value_raw_margin))
    .slice(0, 8)
    .map((item) => ({
      feature: item.feature,
      group: item.feature_group,
      value: Number(item.shap_value_raw_margin.toFixed(3)),
    }));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-slate-100">
            Local Feature Attribution (SHAP Log-Odds Contribution)
          </h4>
          <p className="text-xs text-slate-400">
            Positive values push prediction toward <span className="text-rose-400 font-medium">Pathogenic</span>; negative values toward <span className="text-sky-400 font-medium">Benign</span>.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" /> Pathogenic Driver
          </span>
          <span className="flex items-center gap-1.5 text-sky-400">
            <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" /> Benign Driver
          </span>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sortedData}
            layout="vertical"
            margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
          >
            <XAxis
              type="number"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={165}
              tick={{ fill: '#e2e8f0', fontSize: 11, fontFamily: 'monospace' }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="#64748b" strokeDasharray="3 3" />
            <Tooltip
              cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
              contentStyle={{
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '12px',
              }}
              formatter={(val, _name, props) => {
                const numVal = Number(val ?? 0);
                return [
                  `${numVal > 0 ? '+' : ''}${numVal} (${props?.payload?.group ?? ''})`,
                  'SHAP Margin',
                ];
              }}
            />
            <Bar dataKey="value" radius={[4, 4, 4, 4]} barSize={18}>
              {sortedData.map((entry, idx) => (
                <Cell
                  key={`shap-${idx}`}
                  fill={entry.value >= 0 ? '#f43f5e' : '#0ea5e9'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
import React, { useState } from 'react';
import { Sliders, Database, Compass } from 'lucide-react';
import { BioFeatures, GeometryFeatures } from '@/types';

interface FeatureInspectorTabsProps {
  bioFeatures: BioFeatures;
  geometry: GeometryFeatures;
  onPerturbBio: (patch: Partial<BioFeatures>) => void;
  onPerturbGeom: (patch: Partial<GeometryFeatures>) => void;
  onResetPerturbation: () => void;
  hasPerturbation: boolean;
}

export const FeatureInspectorTabs: React.FC<FeatureInspectorTabsProps> = ({
  bioFeatures,
  geometry,
  onPerturbBio,
  onPerturbGeom,
  onResetPerturbation,
  hasPerturbation,
}) => {
  const [activeTab, setActiveTab] = useState<'bio' | 'geom' | 'whatif'>('bio');

  const bioEntries = Object.entries(bioFeatures);
  const geomEntries = Object.entries(geometry);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('bio')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              activeTab === 'bio'
                ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            <Database className="h-3.5 w-3.5" />
            Bio & SpliceAI (20)
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('geom')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              activeTab === 'geom'
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            <Compass className="h-3.5 w-3.5" />
            Manifold Geometry (8)
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('whatif')}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              activeTab === 'whatif'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            <Sliders className="h-3.5 w-3.5" />
            What-If Perturbation Lab
          </button>
        </div>

        {hasPerturbation && (
          <button
            type="button"
            onClick={onResetPerturbation}
            className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-300 hover:bg-amber-500/20"
          >
            Reset Perturbed Values
          </button>
        )}
      </div>

      {activeTab === 'bio' && (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          {bioEntries.map(([key, val]) => (
            <div
              key={key}
              className="rounded-lg border border-slate-800/90 bg-slate-950/60 px-3 py-2"
            >
              <div className="truncate font-mono text-[11px] text-slate-400" title={key}>
                {key}
              </div>
              <div className="mt-0.5 font-mono text-sm font-semibold text-slate-100">
                {typeof val === 'number' ? Number(val.toPrecision(4)) : val}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'geom' && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {geomEntries.map(([key, val]) => {
            const isDna = key.startsWith('dna_');
            return (
              <div
                key={key}
                className={`rounded-lg border px-3 py-2.5 ${
                  isDna
                    ? 'border-indigo-500/30 bg-indigo-950/20'
                    : 'border-emerald-500/30 bg-emerald-950/20'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-slate-300">{key}</span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                      isDna ? 'bg-indigo-500/20 text-indigo-300' : 'bg-emerald-500/20 text-emerald-300'
                    }`}
                  >
                    {isDna ? 'DNA' : 'PROT'}
                  </span>
                </div>
                <div className="mt-1 font-mono text-base font-bold text-slate-100">
                  {val.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {activeTab === 'whatif' && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-1.5 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-slate-300">phyloP100way_vertebrate</span>
              <span className="font-mono font-semibold text-sky-400">
                {bioFeatures.phyloP100way_vertebrate.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={-5}
              max={10}
              step={0.1}
              value={bioFeatures.phyloP100way_vertebrate}
              onChange={(e) =>
                onPerturbBio({ phyloP100way_vertebrate: parseFloat(e.target.value) })
              }
              className="w-full accent-sky-500"
            />
          </div>

          <div className="space-y-1.5 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-slate-300">SpliceAI_pred_DS_max</span>
              <span className="font-mono font-semibold text-amber-400">
                {bioFeatures.SpliceAI_pred_DS_max.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={bioFeatures.SpliceAI_pred_DS_max}
              onChange={(e) =>
                onPerturbBio({ SpliceAI_pred_DS_max: parseFloat(e.target.value) })
              }
              className="w-full accent-amber-500"
            />
          </div>

          <div className="space-y-1.5 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-slate-300">prot_LVD_Relative</span>
              <span className="font-mono font-semibold text-emerald-400">
                {geometry.prot_LVD_Relative.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={0.8}
              step={0.005}
              value={geometry.prot_LVD_Relative}
              onChange={(e) =>
                onPerturbGeom({ prot_LVD_Relative: parseFloat(e.target.value) })
              }
              className="w-full accent-emerald-500"
            />
          </div>

          <div className="space-y-1.5 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-slate-300">gnomADe_AF</span>
              <span className="font-mono font-semibold text-rose-400">
                {bioFeatures.gnomADe_AF.toFixed(4)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.001}
              value={bioFeatures.gnomADe_AF}
              onChange={(e) => onPerturbBio({ gnomADe_AF: parseFloat(e.target.value) })}
              className="w-full accent-rose-500"
            />
          </div>
        </div>
      )}
    </div>
  );
};
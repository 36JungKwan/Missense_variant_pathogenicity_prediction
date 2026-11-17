import React from 'react';
import { Dna, Layers, Activity, Compass, Lock, RotateCcw } from 'lucide-react';
import { ModalityKey } from '../../types/variant';

interface ModalityTogglePanelProps {
  activeModalities: ModalityKey[];
  onChange: (updated: ModalityKey[]) => void;
  disabled?: boolean;
}

const MODALITY_CONFIG: Array<{
  key: ModalityKey;
  label: string;
  subtitle: string;
  dimInfo: string;
  icon: React.ElementType;
  accentColor: string;
}> = [
  {
    key: 'dna',
    label: 'DNA Embeddings',
    subtitle: 'Nucleotide Transformer v2 (500M)',
    dimInfo: 'ΔE_dna (E_alt - E_ref)',
    icon: Dna,
    accentColor: 'border-indigo-500 bg-indigo-500/10 text-indigo-400',
  },
  {
    key: 'prot',
    label: 'Protein Embeddings',
    subtitle: 'ESM-1b / ESM-1v (650M)',
    dimInfo: 'ΔE_prot (E_alt - E_ref)',
    icon: Layers,
    accentColor: 'border-emerald-500 bg-emerald-500/10 text-emerald-400',
  },
  {
    key: 'bio',
    label: 'Biological & SpliceAI',
    subtitle: '11 Conservation/AF + 9 SpliceAI',
    dimInfo: '20 Tabular Features',
    icon: Activity,
    accentColor: 'border-amber-500 bg-amber-500/10 text-amber-400',
  },
  {
    key: 'geom',
    label: 'Latent Geometry',
    subtitle: 'LVD_L2, Cosine, LID, Relative',
    dimInfo: '8 Manifold Features',
    icon: Compass,
    accentColor: 'border-cyan-500 bg-cyan-500/10 text-cyan-400',
  },
];

export const ModalityTogglePanel: React.FC<ModalityTogglePanelProps> = ({
  activeModalities,
  onChange,
  disabled = false,
}) => {
  const handleToggle = (key: ModalityKey) => {
    if (disabled) return;
    const isActive = activeModalities.includes(key);
    if (isActive) {
      // Bảo vệ: Không cho phép tắt nhánh duy nhất còn lại
      if (activeModalities.length <= 1) return;
      onChange(activeModalities.filter((m) => m !== key));
    } else {
      onChange([...activeModalities, key]);
    }
  };

  const handleResetAll = () => {
    if (!disabled) {
      onChange(['dna', 'prot', 'bio', 'geom']);
    }
  };

  const ablationCode = [...activeModalities].sort().join('_');

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Interactive Modality Ablation (16 Configurations)
          </h3>
          <p className="text-xs text-slate-400">
            Toggle branches dynamically to inspect RB-Net router adaptation and prediction shifts.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 font-mono text-xs text-slate-300">
            Config: <strong className="text-sky-400">{ablationCode}</strong>
          </span>
          {activeModalities.length < 4 && (
            <button
              type="button"
              onClick={handleResetAll}
              disabled={disabled}
              className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 transition hover:bg-slate-700 disabled:opacity-50"
            >
              <RotateCcw className="h-3 w-3" />
              Enable All
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {MODALITY_CONFIG.map((item) => {
          const Icon = item.icon;
          const isActive = activeModalities.includes(item.key);
          const isLocked = isActive && activeModalities.length === 1;

          return (
            <button
              key={item.key}
              type="button"
              onClick={() => handleToggle(item.key)}
              disabled={disabled || isLocked}
              title={
                isLocked
                  ? 'At least one modality must remain active'
                  : `Click to ${isActive ? 'disable' : 'enable'} ${item.label}`
              }
              className={`group relative flex flex-col justify-between rounded-lg border p-3 text-left transition-all ${
                isActive
                  ? `${item.accentColor} shadow-sm`
                  : 'border-slate-800 bg-slate-950/60 text-slate-500 opacity-65 hover:opacity-90'
              } ${isLocked ? 'cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className="text-xs font-semibold tracking-wide text-slate-100">
                    {item.label}
                  </span>
                </div>
                {isLocked ? (
                  <Lock className="h-3.5 w-3.5 text-slate-400" />
                ) : (
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${
                      isActive ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-slate-600'
                    }`}
                  />
                )}
              </div>

              <div className="mt-2.5">
                <p className="text-[11px] text-slate-300/90">{item.subtitle}</p>
                <p className="mt-0.5 font-mono text-[10px] text-slate-400">{item.dimInfo}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
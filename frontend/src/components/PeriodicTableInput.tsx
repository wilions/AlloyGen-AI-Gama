import { useState, useCallback } from 'react';

// Periodic table layout data
const PERIODIC_TABLE: (string | null)[][] = [
  ['H', null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 'He'],
  ['Li', 'Be', null, null, null, null, null, null, null, null, null, null, 'B', 'C', 'N', 'O', 'F', 'Ne'],
  ['Na', 'Mg', null, null, null, null, null, null, null, null, null, null, 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
  ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
  ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
  ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
];

// Elements commonly used in alloys (transition metals, etc.)
const ALLOY_ELEMENTS = new Set([
  'Li', 'Be', 'B', 'C', 'N', 'Al', 'Si', 'P', 'S',
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
  'Ga', 'Ge', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
  'In', 'Sn', 'Sb', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
  'Pb', 'Bi', 'Ce', 'Nd',
]);

// Noble gases / non-metallic — disabled
const DISABLED_ELEMENTS = new Set(['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'At', 'Po', 'Tc']);

interface Props {
  onCompositionChange: (composition: Record<string, number>) => void;
  initialComposition?: Record<string, number>;
}

export default function PeriodicTableInput({ onCompositionChange, initialComposition = {} }: Props) {
  const [selected, setSelected] = useState<Record<string, number>>(initialComposition);

  const total = Object.values(selected).reduce((sum, v) => sum + v, 0);

  const toggleElement = useCallback((el: string) => {
    setSelected(prev => {
      const next = { ...prev };
      if (el in next) {
        delete next[el];
      } else {
        next[el] = 0;
      }
      onCompositionChange(next);
      return next;
    });
  }, [onCompositionChange]);

  const updateValue = useCallback((el: string, value: number) => {
    setSelected(prev => {
      const next = { ...prev, [el]: value };
      onCompositionChange(next);
      return next;
    });
  }, [onCompositionChange]);

  return (
    <div className="rounded-lg bg-gray-900/60 p-3 border border-gray-700/50">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold text-gray-300">Composition Input</h4>
        <span className={`text-xs font-mono ${Math.abs(total - 100) < 0.1 ? 'text-emerald-400' : total > 100 ? 'text-red-400' : 'text-amber-400'}`}>
          Total: {total.toFixed(1)}%
          {Math.abs(total - 100) >= 0.1 && total > 0 && (
            <span className="text-gray-500 ml-1">({total < 100 ? `${(100 - total).toFixed(1)}% remaining` : 'exceeds 100%'})</span>
          )}
        </span>
      </div>

      {/* Periodic table grid */}
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: 'repeat(18, minmax(0, 1fr))' }}>
        {PERIODIC_TABLE.map((row, ri) =>
          row.map((el, ci) => {
            if (el === null) {
              return <div key={`${ri}-${ci}`} />;
            }
            const isDisabled = DISABLED_ELEMENTS.has(el);
            const isSelected = el in selected;
            const isAlloy = ALLOY_ELEMENTS.has(el);

            return (
              <button
                key={el}
                disabled={isDisabled}
                onClick={() => !isDisabled && toggleElement(el)}
                className={`
                  w-full aspect-square text-[9px] font-semibold rounded-sm transition-all
                  flex items-center justify-center
                  ${isDisabled ? 'bg-gray-800/30 text-gray-600 cursor-not-allowed' : ''}
                  ${isSelected ? 'bg-indigo-500/30 border-2 border-indigo-400 text-indigo-300 shadow-sm shadow-indigo-500/20' : ''}
                  ${!isSelected && !isDisabled && isAlloy ? 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 border border-gray-600/30' : ''}
                  ${!isSelected && !isDisabled && !isAlloy ? 'bg-gray-800/30 text-gray-500 hover:bg-gray-700/40 border border-gray-700/20' : ''}
                `}
                title={el}
              >
                {el}
              </button>
            );
          })
        )}
      </div>

      {/* Selected elements with value inputs */}
      {Object.keys(selected).length > 0 && (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {Object.entries(selected).map(([el, val]) => (
            <div key={el} className="flex items-center gap-1.5 bg-gray-800/50 rounded px-2 py-1">
              <span className="text-xs font-semibold text-indigo-400 w-6">{el}</span>
              <input
                type="number"
                min={0}
                max={100}
                step={0.1}
                value={val || ''}
                onChange={(e) => updateValue(el, parseFloat(e.target.value) || 0)}
                className="flex-1 bg-transparent border-b border-gray-600 text-xs text-gray-200 outline-none focus:border-indigo-400 w-12 text-right"
                placeholder="0"
              />
              <span className="text-[10px] text-gray-500">%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import React from 'react';
import { Sun, Snowflake, CloudRain, Flame } from 'lucide-react';

export const SeasonalAnalysis: React.FC = () => {
  const seasons = [
    {
      name: 'Post-Monsoon (Oct – Nov)',
      icon: <Flame className="w-5 h-5 text-rose-400" />,
      color: 'border-rose-900/40 bg-rose-950/20',
      dominant: 'Biomass Burning (Crop Residue Fires)',
      avgPm25: '298.4 µg/m³',
      pattern: 'High sensitivity to upwind stubble burning signals combined with declining transport winds. Model biomass attribution reaches peak annual share (up to 45%).',
      counterfactualSensitivity: 'biomass_low sensitivity averages -28.4 µg/m³ across this regime.',
    },
    {
      name: 'Winter (Dec – Feb)',
      icon: <Snowflake className="w-5 h-5 text-sky-400" />,
      color: 'border-sky-900/40 bg-sky-950/20',
      dominant: 'Wind Ventilation & Thermal Inversion',
      avgPm25: '312.1 µg/m³',
      pattern: 'Low boundary layer height (<400m) and calm surface winds (<10 km/h) trap accumulated pollutants. Stagnation attribution dominates winter peaks.',
      counterfactualSensitivity: 'wind_dispersion sensitivity averages -24.1 µg/m³ across winter episodes.',
    },
    {
      name: 'Summer (Mar – May)',
      icon: <Sun className="w-5 h-5 text-amber-400" />,
      color: 'border-amber-900/40 bg-amber-950/20',
      dominant: 'Meteorology & Mineral Dust',
      avgPm25: '165.2 µg/m³',
      pattern: 'High boundary layer ventilation and thermal convective mixing increase dispersion, offset by dust transport episodes.',
      counterfactualSensitivity: 'meteorology_normal sensitivity averages -12.4 µg/m³.',
    },
    {
      name: 'Monsoon (Jun – Sep)',
      icon: <CloudRain className="w-5 h-5 text-emerald-400" />,
      color: 'border-emerald-900/40 bg-emerald-950/20',
      dominant: 'Meteorological Washout (Precipitation)',
      avgPm25: '58.4 µg/m³',
      pattern: 'Wet deposition and high wind dispersion maintain lowest annual PM2.5 baseline.',
      counterfactualSensitivity: 'Minimal sensitivity to emission features due to dominant rain washout.',
    },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-5">
      <div>
        <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
          Phase 4C Seasonal Attribution Regimes
        </span>
        <h2 className="text-lg font-bold text-slate-100">
          Delhi NCR Seasonal Attribution &amp; Sensitivity Patterns
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {seasons.map((s) => (
          <div
            key={s.name}
            className={`p-4 rounded-xl border ${s.color} space-y-2.5`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-slate-100 flex items-center gap-2">
                {s.icon}
                {s.name}
              </span>
              <span className="text-xs font-mono font-semibold bg-slate-950 px-2.5 py-1 rounded border border-slate-800 text-slate-200">
                Avg PM2.5: {s.avgPm25}
              </span>
            </div>

            <div className="text-xs text-slate-300 font-mono">
              <span className="text-slate-400">Dominant Model Signal:</span>{' '}
              <span className="text-sky-300 font-bold">{s.dominant}</span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              {s.pattern}
            </p>

            <div className="bg-slate-950 p-2.5 rounded border border-slate-800/80 text-[11px] font-mono text-purple-300">
              {s.counterfactualSensitivity}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

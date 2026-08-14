import React from 'react';
import { EnvironmentalValidationResponse } from '../types';
import { ShieldCheck, Wind, Flame, CloudRain } from 'lucide-react';

interface EnvironmentalEvidenceProps {
  validation: EnvironmentalValidationResponse;
}

export const EnvironmentalEvidence: React.FC<EnvironmentalEvidenceProps> = ({ validation }) => {
  const getIcon = (group: string) => {
    switch (group) {
      case 'biomass_burning':
        return <Flame className="w-4 h-4 text-rose-400" />;
      case 'wind_ventilation':
        return <Wind className="w-4 h-4 text-sky-400" />;
      default:
        return <CloudRain className="w-4 h-4 text-purple-400" />;
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Phase 4C Environmental Validation Framework
          </span>
          <h3 className="text-sm font-semibold text-slate-200">
            Independent Observational Indicators for {validation.date}
          </h3>
        </div>

        <span
          className={`px-3 py-1 rounded text-xs font-semibold uppercase border ${
            validation.validation_status === 'WARNING_CONFLICT'
              ? 'bg-rose-950/80 border-rose-800 text-rose-300'
              : 'bg-emerald-950/80 border-emerald-800 text-emerald-300'
          }`}
        >
          Status: {validation.validation_status}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {validation.group_evidence.map((ev) => (
          <div
            key={ev.group_name}
            className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800 flex flex-col justify-between space-y-2"
          >
            <div className="flex items-center justify-between text-xs">
              <span className="flex items-center gap-1.5 font-semibold text-slate-200 capitalize">
                {getIcon(ev.group_name)}
                {ev.group_name.replace('_', ' ')}
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">
                {ev.evidence_status}
              </span>
            </div>

            <div className="text-xs text-slate-400 space-y-1 font-mono">
              <div>
                <span className="text-slate-500">Indicator:</span>{' '}
                <span className="text-slate-300">{ev.supporting_indicator}</span>
              </div>
              <div>
                <span className="text-slate-500">Observed Value:</span>{' '}
                <span className="text-sky-300 font-bold">
                  {ev.observed_value !== null ? ev.observed_value : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Expected Rel:</span>{' '}
                <span className="text-slate-300 capitalize">{ev.relationship}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

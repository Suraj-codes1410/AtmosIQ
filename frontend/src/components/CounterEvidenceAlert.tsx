import React from 'react';
import { CounterEvidenceItem } from '../types';
import { AlertOctagon } from 'lucide-react';

interface CounterEvidenceAlertProps {
  conflicts: CounterEvidenceItem[];
}

export const CounterEvidenceAlert: React.FC<CounterEvidenceAlertProps> = ({ conflicts }) => {
  if (!conflicts || conflicts.length === 0) return null;

  return (
    <div className="bg-rose-950/60 border border-rose-800/80 rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-2 text-rose-300 font-bold text-sm">
        <AlertOctagon className="w-5 h-5 text-rose-400 shrink-0" />
        Counter-Evidence Conflict Detected (Phase 4C Policy Engine)
      </div>
      <p className="text-xs text-rose-200/90 leading-relaxed">
        Independent environmental indicators for this date do <em className="underline">not</em> fully agree with the frozen model's TreeSHAP attribution. AtmosIQ explicitly surfaces contradictory evidence to prevent attribution overconfidence.
      </p>

      <div className="space-y-2">
        {conflicts.map((c, i) => (
          <div
            key={i}
            className="bg-slate-950/80 border border-rose-900/60 rounded p-2.5 text-xs font-mono space-y-1"
          >
            <div className="flex items-center justify-between text-rose-300">
              <span className="font-bold uppercase">Group: {c.group}</span>
              <span className="text-[10px] bg-rose-900/50 text-rose-200 px-2 py-0.5 rounded border border-rose-800">
                Severity: {c.severity}
              </span>
            </div>
            <p className="text-slate-300 text-[11px] font-sans">{c.reason}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

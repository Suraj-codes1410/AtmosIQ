import React from 'react';
import { ConfidenceResponse } from '../types';
import { ShieldCheck, AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';

interface ConfidenceBadgeProps {
  confidence: ConfidenceResponse;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ confidence }) => {
  const getBadgeStyle = (lvl: string) => {
    switch (lvl) {
      case 'HIGH':
        return 'bg-emerald-950/80 border-emerald-800 text-emerald-300';
      case 'MODERATE':
        return 'bg-amber-950/80 border-amber-800 text-amber-300';
      case 'LOW':
        return 'bg-rose-950/80 border-rose-800 text-rose-300';
      default:
        return 'bg-slate-900 border-slate-800 text-slate-400';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
            Attribution Confidence Evaluator
          </span>
          <h3 className="text-sm font-semibold text-slate-200">
            System Confidence Score: <span className="font-mono text-sky-400">{confidence.confidence_score.toFixed(2)}</span>
          </h3>
        </div>

        <span
          className={`px-3 py-1 rounded font-mono font-bold text-xs border ${getBadgeStyle(
            confidence.confidence_level
          )}`}
        >
          Confidence: {confidence.confidence_level}
        </span>
      </div>

      <div className="space-y-2 pt-1">
        {/* Supporting Reasons */}
        {confidence.supporting_reasons.length > 0 && (
          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1">
            <span className="text-[11px] uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1">
              <CheckCircle className="w-3.5 h-3.5" />
              Supporting Evidence Factors
            </span>
            <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
              {confidence.supporting_reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Risk / Conflict Factors */}
        {confidence.risk_factors.length > 0 && (
          <div className="bg-slate-950 p-3 rounded border border-slate-800 space-y-1">
            <span className="text-[11px] uppercase tracking-wider text-amber-400 font-bold flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" />
              Uncertainty &amp; Risk Penalties
            </span>
            <ul className="list-disc list-inside text-xs text-amber-200/90 space-y-1">
              {confidence.risk_factors.map((rf, i) => (
                <li key={i}>{rf}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

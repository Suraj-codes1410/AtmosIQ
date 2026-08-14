import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

export const ScientificDisclaimer: React.FC = () => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-4 text-xs text-slate-300 space-y-2">
      <div className="flex items-center gap-2 text-amber-400 font-semibold uppercase tracking-wider text-[11px]">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        Scientific Limitation & Interpretation Safeguard
      </div>
      <div className="font-mono text-amber-200 bg-amber-950/40 border border-amber-800/40 px-3 py-1.5 rounded text-[11px] leading-relaxed">
        PREDICTIVE IMPORTANCE &ne; SHAP ATTRIBUTION &ne; COUNTERFACTUAL MODEL RESPONSE &ne; CAUSAL EFFECT &ne; ACTUAL EMISSION CONTRIBUTION
      </div>
      <p className="leading-relaxed text-slate-400">
        <strong className="text-slate-200">PM2.5 Persistence Caveat:</strong> Historical PM2.5 features (pm25_persistence group) represent the model's dependence on prior atmospheric pollution state history. It is <em className="text-slate-300">not</em> an independent physical emission source.
      </p>
      <p className="leading-relaxed text-slate-400">
        <strong className="text-slate-200">Counterfactual Sensitivity Caveat:</strong> Counterfactual scenario responses (&Delta;&ycirc;) represent frozen model sensitivities under controlled feature interventions. They do <em className="text-slate-300">not</em> constitute physical chemical-transport simulations or guaranteed emission-reduction outcomes.
      </p>
    </div>
  );
};
